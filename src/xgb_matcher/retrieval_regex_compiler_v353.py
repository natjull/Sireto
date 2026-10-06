"""Exact-language regex compilation repair; historical native recipes stay frozen.

Only a native DFA-size error triggers suffix factoring of literal alternatives.
No pattern/partition is discarded, no candidate pool is created, and native
RegexQuery (including its constant weight) and the single collector are retained.
Unrecognized compilation errors and an unsuccessful repair remain fatal.
"""
import re

from . import retrieval_regional_initial_v286 as original

POLICY = 'V353_EXACT_LITERAL_SUFFIX_FACTORING_ON_NATIVE_DFA_SIZE_ERROR'
SIZE_ERROR = 'RegexQueryError: Compiled regex exceeds size limit of 1000 states'
LITERAL_ALTERNATIVES = re.compile(r'\(([A-Z0-9 ]+(?:\|[A-Z0-9 ]+)*)\)')


def suffix_factor(words):
    """A trie on reversed literal strings, emitted in forward match order.

    Distributivity: (XS | YS) = (X | Y)S. Empty prefixes are retained.
    Spaces are literals, NOT token boundaries invented by this compiler.
    """
    if not words or any(not isinstance(w, str) or re.fullmatch('[A-Z0-9 ]*', w) is None for w in words):
        raise ValueError('Nonempty sequence of ASCII literal alternatives required')
    groups = {}
    empty = False
    for word in words:
        if not word:
            empty = True
        else:
            groups.setdefault(word[-1], []).append(word[:-1])
    parts = [''] if empty else []
    parts.extend(suffix_factor(prefixes) + last for last, prefixes in sorted(groups.items()))
    return parts[0] if len(parts) == 1 else '(' + '|'.join(parts) + ')'


def equivalent_pattern(pattern):
    """Factor only groups containing literal alternatives; preserve operators."""
    # Keep the ORIGINAL group's scope: the following *, ?, + or concatenation
    # must bind the whole factored expression, not its final literal suffix.
    return LITERAL_ALTERNATIVES.sub(lambda m: '(' + suffix_factor(m[1].split('|')) + ')', pattern)


class _Queries:
    def __init__(self, native_query):
        self._native = native_query
        self.repairs = []

    def __getattr__(self, name):
        return getattr(self._native, name)

    def regex_query(self, schema, field_name, regex_pattern):
        try:
            # Successful historical queries are constructed byte-for-byte as
            # before, rather than opportunistically rewriting every regex.
            return self._native.regex_query(schema, field_name, regex_pattern)
        except ValueError as exc:
            if SIZE_ERROR not in str(exc):
                raise
            rewritten = equivalent_pattern(regex_pattern)
            if rewritten == regex_pattern:
                raise
            result = self._native.regex_query(schema, field_name, rewritten)
            self.repairs.append(dict(field=field_name, original=regex_pattern, factored=rewritten))
            return result


class Backend:
    """Forward every native operation except the narrowly repaired compiler."""
    def __init__(self, backend):
        self._backend = backend
        self.queries = _Queries(backend._tantivy.Query)
        self._tantivy = _Native(backend._tantivy, self.queries)

    def __getattr__(self, name):
        return getattr(self._backend, name)


class _Native:
    def __init__(self, native, queries):
        self._native = native
        self.Query = queries

    def __getattr__(self, name):
        return getattr(self._native, name)


def search(backend, *args, **kwargs):
    repaired = Backend(backend)
    found, trace = original.search(repaired, *args, **kwargs)
    trace['regex_compiler_policy'] = POLICY
    trace['regex_patterns_repaired'] = len(repaired.queries.repairs)
    trace['regex_language_changed'] = False
    trace['regex_patterns_discarded'] = 0
    return found, trace
