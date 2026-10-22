"""Exact interval membership for immutable metadata-audit queries.

This accelerates comparisons only: no path, metadata, seed namespace, scope,
classification, reservation or conflict rule is removed or changed.
"""
from bisect import bisect_right
from dataclasses import dataclass

VERSION='warehouse-exact-seed-interval-index.v6.42'


def merged(intervals):
    out=[]
    for low,high in sorted(intervals):
        if out and low<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],high))
        else:out.append((low,high))
    return tuple(out)


def arrays(intervals):
    values=merged(intervals)
    return tuple(p[0] for p in values),tuple(p[1] for p in values)


def contains(value,pair):
    starts,ends=pair
    if not starts or value<starts[0] or value>=ends[-1]:return False
    index=bisect_right(starts,value)-1
    return index>=0 and value<ends[index]


@dataclass(frozen=True)
class SeedIntervalIndex:
    rows:tuple
    union:tuple

    @classmethod
    def compile(cls,families):
        rows=tuple((row['name'],arrays(row['intervals'])) for row in families)
        union=arrays([interval for row in families for interval in row['intervals']])
        return cls(rows,union)

    def matches(self,value):
        if type(value) is not int or not contains(value,self.union):return []
        return [name for name,pair in self.rows if contains(value,pair)]
