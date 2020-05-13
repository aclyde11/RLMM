import sys
from xmlrpc.client import ServerProxy, Binary, Fault

from openeye import oechem


def fastrocs_query(qmol, numHits, host, verbose=False):
    ofs = oechem.oemolostream()
    ofs.SetFormat(oechem.OEFormat_OEB)
    ofs.openstring()
    oechem.OEWriteMolecule(ofs, qmol)
    bytes = ofs.GetString()

    s = ServerProxy("http://" + host)
    data = Binary(bytes)
    idx = s.SubmitQuery(data, numHits)

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
            if verbose:
                print("%s/%s" % ("current", "total"))
            first = False
        if verbose:
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
