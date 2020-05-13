import os
import sys
import argparse
from openeye import oechem
from xmlrpc.client import ServerProxy, Binary, Fault

def GetFormatExtension(fname):
    base, ext = os.path.splitext(fname.lower())
    if ext == ".gz":
        base, ext = os.path.splitext(base)
        ext += ".gz"
    return ext

def main(qmol, numHits, host):
    ofs = oechem.oemolostream()
    ofs.SetFormat(oechem.OEFormat_OEB)
    ofs.openstring()
    oechem.OEWriteMolecule(ofs, qmol)
    bytes = ofs.GetString()
    print(bytes)


    # try:
    #     fh = open(qfname, 'rb')
    # except IOError:
    #     sys.stderr.write("Unable to open '%s' for reading" % qfname)
    #     return 1

    iformat = 'oeb'
    oformat = 'oeb'

    s = ServerProxy("http://" + host)
    data = Binary(bytes)
    exit()
    idx = s.SubmitQuery(data, numHits)
    # except Fault as e:
    #     if "TypeError" in e.faultString:
    #         idx = s.SubmitQuery(data, numHits)
    #     else:
    #         sys.stderr.write(str(e))
    #         return 1

    first = False
    while True:
        blocking = True
        try:
            current, total = s.QueryStatus(idx, blocking)
        except Fault as e:
            print(str(e), file=sys.stderr)
            return 1

        if total == 0:
            continue

        if first:
            print("%s/%s" % ("current", "total"))
            first = False
        print("%i/%i" % (current, total))

        if total <= current:
            break

    results = s.QueryResults(idx)

    ifs = oechem.oemolistream()
    ifs.openstring(results.data)
    ifs.SetFormat(oechem.OEFormat_OEB)
    mols = []
    for mol in ifs.GetOEMols():
        mols.append(oechem.OEMol(mol))


    return mols


if __name__ == '__main__':
    name_mol = sys.argv[1]
    ifs = oechem.oemolistream(name_mol)
    mol = oechem.OEMol()
    oechem.OEReadMolecule(ifs, mol)
    ifs.close()

    host = sys.argv[2]
    main(mol, 10, host)