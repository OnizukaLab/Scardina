import sqlparse as sp


def is_subselect(parsed):
    if not parsed.is_group:
        return False
    for item in parsed.tokens:
        if item.ttype is sp.tokens.DML and item.value.upper() == "SELECT":
            return True
    return False


def extract_from_part(parsed):
    from_seen = False
    for item in parsed.tokens:
        if item.is_whitespace:
            continue
        if from_seen:
            if is_subselect(item):
                for x in extract_from_part(item):
                    yield x
            elif (
                item.ttype is sp.tokens.Keyword
                and "JOIN" not in item.value.upper()
                and "ON" not in item.value.upper()
                and "USING" not in item.value.upper()
            ):
                return
            elif isinstance(item, sp.sql.Identifier):
                yield item
            elif isinstance(item, sp.sql.IdentifierList):
                for i in item.get_identifiers():
                    yield i
        elif item.ttype is sp.tokens.Keyword and item.value.upper() == "FROM":
            from_seen = True


def extract_table_identifiers(token_stream):
    for item in token_stream:
        if isinstance(item, sp.sql.IdentifierList):
            for identifier in item.get_identifiers():
                yield identifier.get_real_name()
                yield identifier.get_name()
        elif isinstance(item, sp.sql.Identifier):
            yield item.get_real_name()
            yield item.get_name()
        # It's a bug to check for Keyword here, but in the example
        # above some tables names are identified as keywords...
        elif item.ttype is sp.tokens.Keyword:
            yield item.value


def extract_alias_to_table(token_stream):
    alias_to_table = {}
    for item in token_stream:
        if isinstance(item, sp.sql.IdentifierList):
            for identifier in item.get_identifiers():
                alias_to_table[
                    identifier.get_name().lower()
                ] = identifier.get_real_name().lower()
        elif isinstance(item, sp.sql.Identifier):
            alias_to_table[item.get_name().lower()] = item.get_real_name().lower()
    return alias_to_table


def extract_where_part(parsed):
    for t in parsed.tokens:
        if isinstance(t, sp.sql.Where):
            return t


def parse_to_conds(sql):
    parsed = sp.parse(sql)[0]
    conds = []

    alias_to_table = extract_alias_to_table(extract_from_part(parsed))
    ident = None
    op = None
    val = None
    is_between = False
    is_join_condition = False
    where_part = extract_where_part(parsed)
    if where_part is None:
        return conds
    for t in where_part.tokens:
        if t.is_whitespace or t.ttype is sp.tokens.Punctuation:
            continue
        if t.ttype is sp.tokens.Keyword and t.value.upper() == "AND" and not is_between:
            if ident.is_group and len(ident.tokens) > 1:
                ident.tokens[0].value = alias_to_table[ident.tokens[0].value]
            if (
                isinstance(val, sp.sql.Identifier)
                and val.is_group
                and val.tokens[1].value != "::"
            ):
                val.tokens[0].value = alias_to_table[val.tokens[0].value.lower()]

            conds.append((op, ident, val, is_join_condition))

            ident = None
            op = None
            val = None
            is_between = False
            is_join_condition = False
        if ident is None:
            if isinstance(t, sp.sql.Identifier):
                ident = t
            elif isinstance(t, sp.sql.Comparison):
                for tt in t.tokens:
                    if tt.is_whitespace:
                        continue
                    if isinstance(tt, sp.sql.Identifier) and (
                        len(tt.tokens) == 1 or tt.tokens[1].normalized != "::"
                    ):
                        if ident is None:
                            ident = tt
                        else:
                            val = tt
                            is_join_condition = True
                    elif isinstance(tt, sp.sql.Identifier):
                        assert (
                            tt.tokens[2].normalized == "TIMESTAMP"
                        ), "only TIMESTAMP is supported (value token with type such as '2000-01-01 00:00:00'::timestamp)"
                        val = tt.tokens[0]
                    elif tt.ttype is sp.tokens.Comparison:
                        op = tt
                    elif isinstance(tt, sp.sql.Parenthesis):
                        inner = tt.tokens[1]
                        if isinstance(inner, sp.sql.IdentifierList):
                            val = list(inner.get_identifiers())
                        else:
                            val = [inner]
                    else:
                        val = tt
        else:
            if op is None:
                if t.ttype is sp.tokens.Keyword:
                    op = t
                    if t.value.upper() == "BETWEEN":
                        is_between = True
            else:
                if t.ttype is sp.tokens.Keyword and t.value.upper() == "NOT NULL":
                    # hacky: IS NOT as an operator. Must refer op.value
                    op.value = "IS NOT"
                    t.value = "NULL"

                if not is_between:
                    val = t
                else:
                    if val is None:
                        val = [t]
                    elif t.ttype is not sp.tokens.Keyword:
                        val.append(t)
                        is_between = False

    if ident.is_group and len(ident.tokens) > 1:
        ident.tokens[0].value = alias_to_table[ident.tokens[0].value]
    if (
        isinstance(val, sp.sql.Identifier)
        and val.is_group
        and (len(tt.tokens) == 1 or val.tokens[1].value != "::")
    ):
        val.tokens[0].value = alias_to_table[val.tokens[0].value.lower()]
    conds.append((op, ident, val, is_join_condition))

    return conds


if __name__ == "__main__":
    sql = """
    select K.a,K.b from (select H.b from (select G.c from (select F.d from
    (select E.e from Alpha as A, B, C, D, E), F), G), H), I, J, K order by 1,2;
    """
    sql = "select * from t left join u as foo on t.id = u.t_id where u.age > 20"
    sql = """
    SELECT COUNT(*) FROM comp_cast_type AS cct1,company_name AS cn,company_type AS ct,complete_cast AS cc,keyword AS k,link_type AS lt,movie_companies AS mc,movie_info AS mi,movie_keyword AS mk,movie_link AS ml,title AS t
    WHERE t.production_year BETWEEN 1950 AND 2000 AND mc.note IS NULL AND cct1.kind IN ('cast', 'crew') AND k.keyword = 'sequel' AND cn.country_code != '[pl]' AND ct.kind = 'production companies' AND lt.link LIKE '%follow%' AND mi.info IN ('Sweden', 'Germany', 'Swedish', 'German') AND lt.id = ml.link_type_id AND ml.movie_id = t.id AND ml.movie_id = mk.movie_id AND ml.movie_id = mc.movie_id AND ml.movie_id = mi.movie_id AND ml.movie_id = cc.movie_id AND t.id = mk.movie_id AND t.id = mc.movie_id AND t.id = mi.movie_id AND t.id = cc.movie_id AND mk.keyword_id = k.id AND mk.movie_id = mc.movie_id AND mk.movie_id = mi.movie_id AND mk.movie_id = cc.movie_id AND mc.company_type_id = ct.id AND mc.company_id = cn.id AND mc.movie_id = mi.movie_id AND mc.movie_id = cc.movie_id AND mi.movie_id = cc.movie_id AND cc.subject_id = cct1.id order
    """
    # sql = """
    # SELECT COUNT(*) FROM company_name AS cn,company_type AS ct,complete_cast AS cc,keyword AS k,kind_type AS kt,movie_companies AS mc,movie_info AS mi,movie_keyword AS mk,title AS t WHERE t.production_year > 2000 AND cn.country_code = '[us]' AND kt.kind IN ('movie') AND kt.dummy IN ('a', 'b') AND mi.note LIKE '%internet%' AND mi.info IS NOT NULL AND kt.id = t.kind_id AND t.id = mi.movie_id AND t.id = mk.movie_id AND t.id = mc.movie_id AND t.id = cc.movie_id AND mi.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id AND mi.movie_id = cc.movie_id AND mk.movie_id = mc.movie_id AND mk.movie_id = cc.movie_id AND mk.keyword_id = k.id AND mc.movie_id = cc.movie_id AND mc.company_id = cn.id AND mc.company_type_id = ct.id;
    # """
    # sql = """
    # select count(1) from dmv where "Revocation Indicator = 'N' and Record Type" = 'VEH ' and "County" = 'SUFFOLK     ' and "Suspension Indicator" = 'N' and "State" <= 'NY' and "Reg Valid Date" <= 2018-09-26T00:00:00.000000000 and "Color" <= 'RD   ' and "Registration Class" >= 'PAS';
    # """

    conds = parse_to_conds(sql)
    for cond in conds:
        print(cond)