import sys
from pathlib import Path

# Allow importing Protenix utilities
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / 'third_parties' / 'Protenix'))

from protenix.openfold_local.data.parsers import convert_stockholm_to_a3m


def main():
    if len(sys.argv) != 3:
        sys.exit('Usage: sto_to_a3m.py <input.sto> <output.a3m>')
    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])
    sto = inp.read_text()
    a3m = convert_stockholm_to_a3m(sto)
    out.write_text(a3m)


if __name__ == '__main__':
    main()

