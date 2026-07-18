import os
import re

HERE = os.path.dirname(__file__)
T = open(os.path.join(HERE, "..", "..", "WRITING_REPORT", "latency.tex"), encoding="utf-8").read()

DASHES = "–—‒―"
print("long dashes:", sum(1 for c in T if c in DASHES))
print("non-ascii  :", sorted({hex(ord(c)) for c in T if ord(c) > 127}))

b = len(re.findall(r"\\begin\{", T))
e = len(re.findall(r"\\end\{", T))
print("begin/end  :", b, e, "OK" if b == e else "MISMATCH")
for env in ["table", "tabular", "figure", "document"]:
    bb = len(re.findall(r"\\begin\{" + env + r"\}", T))
    ee = len(re.findall(r"\\end\{" + env + r"\}", T))
    print("  %-9s %d/%d %s" % (env, bb, ee, "OK" if bb == ee else "MISMATCH"))

print("brace bal  :", T.count("{") - T.count("}"))
print("first-person:", re.findall(r"\b(?:we|our|us|my|ours|I)\b", T))

ROWEND = "\\\\"
for m in re.finditer(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", T, re.S):
    spec, body = m.group(1), m.group(2)
    ncol = sum(1 for c in spec if c in "lcr")
    for line in body.splitlines():
        line = line.strip()
        if line.endswith(ROWEND) and "&" in line:
            amps = line.count("&")
            if amps != ncol - 1:
                print("  ROW MISMATCH (%d cols, %d amps):" % (ncol, amps), line[:70])
print("column-count check done")
