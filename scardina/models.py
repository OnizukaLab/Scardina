import re
import random
from typing import List, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


act_funcs: Dict[str, nn.Module] = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
}


class EmbModule(nn.ModuleList):
    def __init__(
        self,
        input_bins: List[Tuple[str, int]],
        d_word: int,
        cont_fanout: bool,
        smaller_emb: str = "",
    ):
        super().__init__()

        self.input_bins = input_bins
        self.dom_sizes = [size for _, size in input_bins]
        self.d_word = d_word
        self.cont_fanout = cont_fanout
        self.smaller_emb = smaller_emb

        # NOTE: ad-hoc
        def _is_continuous(name: str) -> bool:
            if "_fanout__" in name:
                return True
            if name.endswith("_year"):
                return True
            return False

        self.idxes = [0]
        self.input_idxes = [0]  # WIP
        self.dims = []
        self.sum_of_input_dims = 0
        for i in range(len(self.input_bins)):
            if self.dom_sizes[i] > 1:  # categorical
                if "bound" in self.smaller_emb and "continuous" in self.smaller_emb:
                    d_word = 1 if _is_continuous(self.input_bins[i][0]) else self.d_word
                    self.dims.append(min(d_word, self.dom_sizes[i]))
                elif "bound" in self.smaller_emb:
                    # up to domain size
                    self.dims.append(min(self.d_word, self.dom_sizes[i]))
                elif "continuous" in self.smaller_emb:
                    # embed continuous attrs into 1-dim
                    d_word = (
                        1
                        if _is_continuous(self.input_bins[i][0])
                        else self.dom_sizes[i]
                    )
                    self.dims.append(d_word)
                else:
                    # d_word for all
                    self.dims.append(self.d_word)

                self.append(
                    nn.Embedding(self.dom_sizes[i], self.dims[-1], padding_idx=0)
                )
                self[-1].name = self.input_bins[i][0]
                self.idxes.append(self.idxes[-1] + self.dims[-1])
            else:  # continuous (currently, fanout only)
                self.dims.append(2)
                self.append(nn.Identity())  # dummy
                self.idxes.append(self.idxes[-1] + 2)
        for emb in self:
            if isinstance(emb, nn.Embedding):
                nn.init.normal_(emb.weight, std=0.02)

        # self.unk_embs = nn.ParameterList()
        # for dim in self.dims:
        #     self.unk_embs.append(
        #         nn.Parameter(
        #             torch.zeros(
        #                 dim,
        #             )
        #         )
        #     )

        self.sum_of_dims = self.idxes[-1]
        del self.idxes[-1]
        del self.input_idxes[-1]  # WIP

    def encode(self, data: torch.Tensor, i: int, out: torch.Tensor) -> torch.Tensor:
        """
        for inference (allowing unk)
        """
        datum = data[:, i]
        # FIXME: unk_emb is not working
        if datum is None:
            # [bs, d_word]
            out[:, i] = self.unk_embs[i].unsqueeze(0).expand(out.shape[0], -1)
        else:
            le = self.idxes[i]
            ri = le + self.dims[i]

            if self.dom_sizes[i] > 1:
                out[:, le:ri] = self[i](datum.long())
            else:
                out[:, le:ri] = torch.cat(
                    [
                        torch.ones(
                            (datum.size(0), 1), device=datum.device
                        ),  # mask flag
                        datum.unsqueeze(1).float(),  # value
                    ],
                    1,
                )

    def encode_all(self, data: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """
        for inference (allowing unk)
        """
        for i, datum in enumerate(data.t()):
            self.encode(data, i, out)

    # def encode_wo_unk(self, data, i, out):
    #     datum = data[:, i]
    #     l = self.idxes[i]
    #     r = l + self.dims[i]

    #     if self.dom_sizes[i] > 1:
    #         out[:, l:r] = self[i](datum)
    #     else:
    #         out[:, l:r] = datum.unsqueeze(1)

    def encode_all_wo_unk(self, data: torch.Tensor) -> torch.Tensor:
        """
        for training (not containing unk)
        """
        ys = []
        for i, emb in enumerate(self):
            datum = data[:, i]
            if isinstance(emb, nn.Embedding):
                ys.append(emb(datum.long()))
            else:
                if datum[0] > 0:  # NOTE: check only head
                    # not masked
                    ys.append(torch.ones((datum.size(0), 1), device=datum.device))
                    ys.append(datum.view(-1, 1))
                else:
                    # masked
                    ys.append(torch.zeros((datum.size(0), 2), device=datum.device))

        return torch.cat(ys, 1)  # [bs, concat_dims]

    def decode_as_raw_val(self, logits: torch.Tensor, scol_idx: int) -> torch.Tensor:
        assert self.dom_sizes[scol_idx] == 1

        le = self.idxes[scol_idx]
        ri = le + self.dims[scol_idx]
        return logits[:, le:ri]

    def decode_as_logit(self, logits: torch.Tensor, scol_idx: int) -> torch.Tensor:
        """Returns the logits (vector) corresponding to log p(x_i | x_(<i)).

        Args:
            idx: int, in natural (table) ordering.
            logits: [batch size, ncols+1, d_word].

        Returns:
            logits_for_scol: [batch size, domain size for column idx].
        """
        assert self.dom_sizes[scol_idx] > 1

        le = self.idxes[scol_idx]
        ri = le + self.dims[scol_idx]
        return torch.matmul(
            logits[:, le:ri],
            self[scol_idx].weight.t(),
        )


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_word: int, n_heads: int):
        assert d_word % n_heads == 0

        super(MultiHeadSelfAttention, self).__init__()

        self.d_word = d_word
        self.num_heads = n_heads
        self.d_state = d_word // n_heads

        self.qkv_linear = nn.Linear(d_word, self.d_state * 3 * n_heads, bias=False)
        self.linear = nn.Linear(n_heads * self.d_state, d_word)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # Each input has shape [bs, num cols, d_state * num_heads].
        *start, m = x.size()
        x = x.view(start + [self.num_heads, m // self.num_heads])
        return x.permute(0, 2, 1, 3)

    def _do_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Accepts Q,K,V each shaped [bs, num heads, num cols, d_state].

        Returns transformed [bs, num_heads, num cols, d_state].
        """
        d_k = query.size()[-1]
        scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(d_k)
        attn_weights = F.softmax(scores, dim=-1)  # no mask for non-autoregressive
        out = torch.matmul(attn_weights, value)
        return out

    def forward(
        self, x: torch.Tensor, query_input: torch.Tensor = None
    ) -> torch.Tensor:
        """x: [bs, num cols, d_word].  Output has the same shape."""
        assert x.dim() == 3, x.size()
        bs, ncols, _ = x.size()

        # [bs, num cols, d_state * 3 * num_heads]
        qkv = self.qkv_linear(x)
        # [bs, num heads, num cols, d_state] each
        qs, ks, vs = map(self._split_heads, torch.chunk(qkv, 3, dim=-1))

        if query_input is not None:
            # TODO: obviously can avoid redundant calc.
            qkv = self.qkv_linear(query_input)
            qs, _, _ = map(self._split_heads, torch.chunk(qkv, 3, dim=-1))

        # [bs, num heads, num cols, d_state]
        x = self._do_attention(qs, ks, vs)

        # [bs, num cols, num heads, d_state]
        x = x.transpose(1, 2)
        # Concat all heads' outputs: [bs, num cols, num heads * d_state]
        x = x.contiguous().view(bs, ncols, -1)
        # Then do a transform: [bs, num cols, d_word].
        x = self.linear(x)
        return x


class Block(nn.Module):
    def __init__(self, d_word, d_ff, n_heads, act_func, do_residual=True):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_word, d_ff),
            act_func,
            nn.Linear(d_ff, d_word),
        )
        self.norm1 = nn.LayerNorm(d_word)
        self.norm2 = nn.LayerNorm(d_word)
        self.attn = MultiHeadSelfAttention(d_word, n_heads)
        self.do_residual = do_residual

    def forward(
        self, x: torch.Tensor, query_input: torch.Tensor = None
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.attn(x, query_input=query_input)
        if self.do_residual:
            x += residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        if self.do_residual:
            x += residual

        return x


class NAR(pl.LightningModule):
    def __init__(
        self,
        params: Dict[str, Union[str, int]],
        table_name: str,
        input_bins: List[Tuple[str, int]],
    ):
        super().__init__()

        assert params["act_func_name"].lower() in act_funcs

        self.dataset_name = params["static"]["dataset_name"]
        self.table_name = table_name
        self.center_table_name = re.sub(":.+$", "", self.table_name)
        self.identity = params["static"]["id"]
        self.input_bins = input_bins
        self.emb_sizes = [size for _, size in input_bins]
        # TODO: Use central fanout check
        not_for_precondition_mask = np.array(
            ["fanout" in name for name, _ in input_bins]
        )
        self.not_for_precondition = np.where(not_for_precondition_mask)[0]
        self.precondition_candidates_map = np.where(~not_for_precondition_mask)[0]
        self.n_cols = len(self.emb_sizes)
        self.d_word = params["d_word"]
        self.smaller_emb = params["static"]["smaller_emb"]
        self.act_func = act_funcs[params["act_func_name"].lower()]
        self.cont_fanout = params["static"]["cont_fanout"]
        self.lr = params["lr"]
        self.warmups = params["warmups"]
        assert (self.lr is not None and self.warmups is None) or (
            self.lr is None and self.warmups is not None
        )
        self.max_epochs = params["static"]["n_epochs"]

        self.embs = EmbModule(
            self.input_bins, self.d_word, self.cont_fanout, self.smaller_emb
        )
        self._random_state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def forward_w_encoded(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def nll_masked(
        self, logits: torch.Tensor, label: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        nlls = torch.zeros(logits.size(0), device=logits.device)
        for i in range(self.n_cols):
            if ~mask[i]:
                # skip not masked cols' nll
                continue

            if self.embs.dom_sizes[i] > 1:
                logit = self.embs.decode_as_logit(logits, i)
                nlls += F.cross_entropy(logit, label[:, i].long(), reduction="none")
            else:
                flag_and_fanout = self.embs.decode_as_raw_val(logits, i)
                nlls += F.mse_loss(flag_and_fanout[:, 1], label[:, i], reduction="none")
        nlls /= mask.sum()
        return nlls

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x = batch.to(self.device, non_blocking=True).to(torch.float32)

        # True maens it will be masked (masked attributes will be estimated)
        n_masks = random.choice(range(1, self.n_cols))
        mask_idxes = random.sample(range(self.n_cols), n_masks)
        mask = torch.zeros((self.n_cols,), dtype=torch.bool, device=self.device)
        mask[mask_idxes] = True

        masked_x = x.clone() * ~mask

        y = self.forward(masked_x)
        loss = self.nll_masked(y, x, mask).mean()
        self.log(f"{self.table_name}/tra_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        if batch_idx == 0:
            self._random_state = random.getstate()
        random.seed(batch_idx)  # fix random state while only validation steps

        x = batch.to(self.device, non_blocking=True).to(torch.float32)

        micro_xs = x.split(x.size(0) // 10)
        micro_batch_len = len(micro_xs)
        losses = torch.empty(micro_batch_len, device=self.device)
        weights = torch.empty(micro_batch_len, device=self.device)
        for i, micro_x in enumerate(micro_xs):
            # True maens it will be masked (masked attributes will be estimated)
            n_masks = random.choice(range(1, self.n_cols))
            mask_idxes = random.sample(range(self.n_cols), n_masks)
            mask = torch.zeros((self.n_cols,), dtype=torch.bool, device=self.device)
            mask[mask_idxes] = True

            masked_x = micro_x.clone()
            masked_x *= ~mask

            y = self.forward(masked_x)
            losses[i] = self.nll_masked(y, micro_x, mask).mean()
            weights[i] = micro_x.size(0)

        loss = (losses * weights).sum() / weights.sum()
        return loss

    def validation_step_end(self, val_step_outputs):
        # rewind random state
        random.setstate(self._random_state)

        self.log(
            "epoch", self.current_epoch + 1
        )  # for ASHA scheduler w/ val_check_interval != 1.0
        self.log(f"{self.table_name}/val_loss", val_step_outputs.mean())

    def set_dp(
        self,
        params_to_optimize,
        carrier_params,
        each_epoch_func,
        get_after_backward_func,
    ) -> None:
        self.params_to_optimize = params_to_optimize
        self.carrier_params = carrier_params
        self.each_epoch_func = each_epoch_func
        self.get_after_backward_func = get_after_backward_func

        self.each_epoch_func()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return (
            torch.optim.Adam(self.parameters(), lr=self.lr)
            if self.lr is not None
            else torch.optim.Adam(self.parameters())
        )

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        if self.warmups:
            step = self.trainer.global_step + 1
            # original: (self.d_word ** -0.5) * min(step ** -0.5, step * self.warmups ** -1.5)
            lr = (self.d_word**-1) * min(step**-0.5, step * self.warmups**-1.5)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        optimizer.step(closure=optimizer_closure)


class NARTransformer(NAR):
    def __init__(
        self,
        params: Dict[str, Union[str, int]],
        table_name: str,
        input_bins: List[Tuple[str, int]],
    ):
        super().__init__(params, table_name, input_bins)

        self.name = f"nar-trm-{self.dataset_name}-{self.table_name}-{self.identity}"

        self.n_blocks = params["n_blocks"]
        self.d_ff = params["d_ff"]
        self.n_heads = params["n_heads"]

        self.blocks = nn.Sequential(
            *[
                Block(
                    self.d_word,
                    self.d_ff,
                    self.n_heads,
                    self.act_func,
                    do_residual=True,
                )
                for i in range(self.n_blocks)
            ]
        )
        self.norm = nn.LayerNorm(self.d_word)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch_size, n_scols] -> [batch_size, n_scols, d_word]
        x = self.embs.encode_all_wo_unk(x)
        x = self.blocks(x.view(-1, self.n_cols, self.d_word))
        x = self.norm(x)
        return x.view(x.size(0), -1)

    def forward_w_encoded(self, x: torch.Tensor) -> torch.Tensor:
        # [batch_size, n_scols * d_word] -> [n_scols, n_scols, d_word]
        x = self.blocks(x.view(-1, self.n_cols, self.d_word))
        x = self.norm(x)
        return x.view(x.size(0), -1)


class NARMLP(NAR):
    def __init__(
        self,
        params: Dict[str, Union[str, int]],
        table_name: str,
        input_bins: List[Tuple[str, int]],
    ):
        super().__init__(params, table_name, input_bins)

        self.name = f"nar-mlp-{self.dataset_name}-{self.table_name}-{self.identity}"

        self.d_ff = params["d_ff"]
        self.n_ff = params["n_ff"]

        self.mlp = nn.Sequential(
            nn.Linear(self.embs.sum_of_dims, self.d_ff),
            self.act_func,
            *[
                nn.Sequential(
                    nn.Linear(self.d_ff, self.d_ff),
                    self.act_func,
                )
                for i in range(self.n_ff)
            ],
            nn.Linear(self.d_ff, self.embs.sum_of_dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embs.encode_all_wo_unk(x)
        x = self.mlp(x)
        return x

    def forward_w_encoded(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x
