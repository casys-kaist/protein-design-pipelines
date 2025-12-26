#!/usr/bin/env python
"""
validate_contig.py  —  Quick sanity-check for RFdiffusion contig strings

usage:
  python validate_contig.py 1SHG.pdb "[5-10/A7-12/3-6/A15-22/...]" 
  python validate_contig.py 1SHG.cif "[5-10/A7-12/3-6/A15-22/...]" 
"""

import math
import re, sys, textwrap
from pathlib import Path
from Bio.PDB import PDBParser, MMCIFParser, DSSP
from Bio.PDB.DSSP import residue_max_acc

# ─────────────────────────────────────────────────────────────
def parse_contig(contig: str):
    parsed = []
    for tok in contig.strip("[]").split("/"):
        if re.fullmatch(r"\d+-\d+", tok):
            parsed.append({"type": "var", "range": tuple(map(int, tok.split("-")))})
        elif re.fullmatch(r"[A-Za-z]\d+-\d+", tok):
            chain, rng = tok[0], tuple(map(int, tok[1:].split("-")))
            parsed.append({"type": "fix", "chain": chain, "range": rng})
        elif tok == "0":
            parsed.append({"type": "break"})
    return parsed

def load_structure(fname: str):
    suffix = Path(fname).suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True).get_structure("S", fname)[0]
    return PDBParser(QUIET=True).get_structure("S", fname)[0]

def dssp_map(structure, fname: str):
    try:
        dssp = DSSP(structure, fname, dssp='mkdssp')
        out = {}
        for (chain, resid), prop in dssp.property_dict.items():
            aa = prop[1]
            ss_code = prop[2]
            acc = prop[3]
            if acc is None:
                rsa = float('nan')
            elif acc <= 1.0 + 1e-6:
                rsa = float(acc)
            elif acc <= 100.0 + 1e-6:
                rsa = float(acc) / 100.0
            else:
                rsa = float(acc) / float(residue_max_acc.get(aa, 200.0))
            out[(chain, resid)] = (ss_code, rsa)
        return out
    except Exception as e:
        print(f"Error: {e}")
        return {}

# ─────────────────────────────────────────────────────────────
def main(pdbfile: str, contig: str):
    st   = load_structure(pdbfile)
    dssp = dssp_map(st, pdbfile)
    ok   = True

    for seg in parse_contig(contig):
        if seg["type"] != "fix": continue
        c, (s, e) = seg["chain"], seg["range"]
        if c not in st:
            print(f"[ERROR] Chain '{c}' not in structure."); ok = False; continue
        for r in range(s, e + 1):
            if (" ", r, " ") not in st[c]:
                print(f"[ERROR] Residue {c}{r} missing."); ok = False

    if ok:
        print("✓ All fixed residues exist in structure")

    if dssp:
        print("\nSecondary-structure / RSA snapshot:")
        for seg in parse_contig(contig):
            if seg["type"] != "fix": continue
            c, (s, e) = seg["chain"], seg["range"]
            ann = [dssp.get((c, (" ", r, " ")), ("?", float('nan'))) for r in range(s, e + 1)]
            ss_str  = "".join(ss for ss, _ in ann)
            rsa_vals = [rsa for _, rsa in ann if math.isfinite(rsa)]
            if rsa_vals:
                rsa_avg = sum(rsa_vals) / len(rsa_vals)
                rsa_min = min(rsa_vals)
                rsa_max = max(rsa_vals)
            else:
                rsa_avg = rsa_min = rsa_max = float('nan')
            print(
                f"  {c}{s}-{e}: SS={ss_str[:20]}..., ⟨RSA⟩={rsa_avg:.2f}  "
                f"(min={rsa_min:.2f}, max={rsa_max:.2f})"
            )
    else:
        print("(mkdssp not found or failed — skipping SS/RSA check)")

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(textwrap.dedent(__doc__))
    main(sys.argv[1], sys.argv[2])
