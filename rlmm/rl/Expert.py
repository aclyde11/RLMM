from openeye import oechem, oedocking, oeshape, oeomega, oemolprop
import numpy as np
import subprocess
from simtk import unit


class FastRocsPolicy:

    def __init__(self, env, header='python /vol/ml/aclyde/fastrocs/ShapeDatabaseClient.py dgx1:8080', hits=100, step_size=3.5, orig_pdb=None):
        self.env = env
        self.header = header
        self.hits = hits
        self.step_size = step_size
        self.orig_pdb = orig_pdb
        self.start_receptor = None
        self.start_dobj = None
        if self.orig_pdb is not None:
            pdb = oechem.OEMol()
            prot = oechem.OEMol()
            lig = oechem.OEMol()
            wat = oechem.OEGraphMol()
            other = oechem.OEGraphMol()
            ifs = oechem.oemolistream(self.orig_pdb)
            oechem.OEReadMolecule(ifs, pdb)
            ifs.close()
            if not oechem.OESplitMolComplex(lig, prot, wat, other, pdb):
                print("crap")
            self.start_receptor = oechem.OEGraphMol()
            oedocking.OEMakeReceptor(self.start_receptor, prot, lig)
            self.start_dobj = oedocking.OEDock(oedocking.OEDockMethod_Chemgauss4)
            self.start_dobj.Initialize(self.start_receptor)
            self.start_dobj.DockMultiConformerMolecule()


    def getscores(self,actions, gsmis, prot, ligand, num_returns = 10, return_docked_pose=False):
        if num_returns <= 0:
            num_returns = len(actions)-1
        print("Action space is ", len(actions))
        idxs = list(range(len(actions), min(num_returns,len(actions) - 1)))

        protein = oechem.OEMol(prot)
        receptor = oechem.OEGraphMol()
        oedocking.OEMakeReceptor(receptor, protein, ligand)
        dockobj = oedocking.OEDock(oedocking.OEDockMethod_Chemgauss4)
        dockobj.Initialize(receptor)

        scores = []
        data = []
        for idx in idxs:
            try:
                new_mol, new_mol2, gs, action = actions[idx], actions[idx], gsmis[idx], gsmis[idx]
                dockedpose = oechem.OEMol()
                dockobj.DockMultiConformerMolecule(dockedpose, new_mol, 1)
                ds = dockedpose.GetEnergy()
                ps = dockobj.ScoreLigand(new_mol)
                ds2=None
                if self.start_dobj is not None:
                    dockedpose2 = oechem.OEMol()
                    newmol2 = oechem.OEMol(new_mol)
                    self.start_dobj.DockMultiConformerMolecule(dockedpose2, newmol2, 1)
                    ds2 = dockedpose2.GetEnergy()
                print("SCORE",ds ,ps ,ds2)
                if return_docked_pose:
                    new_mol = oechem.OEMol(dockedpose)
                    new_mol2 = oechem.OEMol(dockedpose)
                oechem.OEAddExplicitHydrogens(new_mol2)
                oechem.OEAddExplicitHydrogens(new_mol)
                data.append((new_mol, new_mol2, gs, action))
                scores.append(ds)
            except:
                continue
        order = np.argsort(scores)
        data = [data[i] for i in order]
        return data

    def choose_action(self):
        self.env.openmm_simulation.get_pdb("test.pdb")
        pdb = oechem.OEMol()
        prot = oechem.OEMol()
        lig = oechem.OEMol()
        wat = oechem.OEGraphMol()
        other = oechem.OEGraphMol()
        ifs = oechem.oemolistream("test.pdb")
        oechem.OEReadMolecule(ifs, pdb)
        ifs.close()
        if not oechem.OESplitMolComplex(lig, prot, wat, other, pdb):
            print("crap")
            exit()

        ofs = oechem.oemolostream("rocs.sdf")
        oechem.OEWriteMolecule(ofs, lig)
        ofs.close()
        subprocess.run(self.header + " rocs.sdf rocshits.sdf " + str(int(self.hits)), shell=True)
        ifs = oechem.oemolistream('rocshits.sdf')
        mols = []
        smiles=[]
        for mol in ifs.GetOEMols():
            mols.append(oechem.OEMol(mol))
            smi = oechem.OEMolToSmiles(mol)
            print("ROCSHIT", smi)
            smiles.append(smi)
        ifs.close()

        data = self.getscores(mols, smiles, prot, lig, num_returns=-1, return_docked_pose=False)
        not_worked = True
        idxs = list(range(len(data)))
        idx = idxs.pop(0)
        counter = 0
        while not_worked:
            try:
                new_mol, new_mol2, gs, action = data[idx]
                self.env.systemloader.reload_system(gs, new_mol, "test.pdb")
                self.env.openmm_simulation = self.env.config.openmmWrapper.get_obj(self.env.systemloader, ln=self.env.systemloader, stepSize=self.step_size * unit.femtoseconds)

                not_worked = False
            except Exception as e:
                print(e)
                if len(idxs) == 0:
                    print("mega fail")
                    exit()
                idx = idxs.pop(0)
        return new_mol2, action

class RandomPolicy:

    def __init__(self, env, return_docked_pose=False, num_returns=-1, step_size=3.5):
        self.return_docked_pose= return_docked_pose
        self.num_returns = num_returns
        self.env = env
        self.step_size = step_size

    def getscores(self,actions, gsmis, prot, num_returns = 10, return_docked_pose=False):
        if num_returns <= 0:
            num_returns = len(actions)-1
        print("Action space is ", len(actions))
        idxs = list(np.random.choice(len(actions), min(num_returns,len(actions) - 1), replace=False).flatten())

        data = []
        for idx in idxs:
            try:
                new_mol, new_mol2, gs, action = self.env.action.get_aligned_action(actions[idx], gsmis[idx])
                data.append((new_mol, new_mol2, gs, action))
            except:
                continue
        return data

    def choose_action(self):
        self.env.openmm_simulation.get_pdb("test.pdb")
        pdb = oechem.OEMol()
        prot = oechem.OEMol()
        lig = oechem.OEMol()
        wat = oechem.OEGraphMol()
        other = oechem.OEGraphMol()
        ifs = oechem.oemolistream("test.pdb")
        oechem.OEReadMolecule(ifs, pdb)
        ifs.close()
        if not oechem.OESplitMolComplex(lig, prot, wat, other, pdb):
            print("crap")
            exit()

        self.env.action.update_mol_aligneer(lig)
        actions, gsmis = self.env.action.get_new_action_set()
        data = self.getscores(actions, gsmis, prot, num_returns=self.num_returns, return_docked_pose=self.return_docked_pose)
        not_worked = True
        idxs = list(range(len(data)))
        idx = idxs.pop(0)
        counter = 0
        while not_worked:
            try:
                new_mol, new_mol2, gs, action = data[idx]
                self.env.systemloader.reload_system(gs, new_mol, "test.pdb")
                self.env.openmm_simulation = self.env.config.openmmWrapper.get_obj(self.env.systemloader, ln=self.env.systemloader, stepSize=self.step_size * unit.femtoseconds, prior_sim=self.env.openmm_simulation.simulation)
                not_worked = False
            except Exception as e:
                print(e)
                if len(idxs) == 0:
                    print("mega fail")
                    exit()
                idx = idxs.pop(0)
        self.env.action.apply_action(new_mol2, action)

        return new_mol2, action

class ExpertPolicy:

    def __init__(self, env, return_docked_pose=False, num_returns=-1, step_size=3.5):
        self.return_docked_pose= return_docked_pose
        self.num_returns = num_returns
        self.env = env
        self.step_size = step_size



    def getscores(self,actions, gsmis, prot, num_returns = 10, return_docked_pose=False):
        if num_returns <= 0:
            num_returns = len(actions)-1
        print("Action space is ", len(actions))
        idxs = list(np.random.choice(len(actions), min(num_returns,len(actions) - 1), replace=False).flatten())

        protein = oechem.OEMol(prot)
        receptor = oechem.OEGraphMol()
        ligand_index = list(set(self.env.openmm_simulation.config.systemloader.get_selection_ligand()))
        pos = np.mean(np.array(self.env.openmm_simulation.get_coordinates()[ligand_index], dtype=np.float32), axis=0) * 10

        print(pos)
        oedocking.OEMakeReceptor(receptor, protein, float(pos[0]), float(pos[1]), float(pos[2]), True)
        dockobj = oedocking.OEDock(oedocking.OEDockMethod_Chemgauss4)
        dockobj.Initialize(receptor)
        pscores = []

        scores = []
        data = []
        for idx in idxs:
            try:
                new_mol, new_mol2, gs, action = self.env.action.get_aligned_action(actions[idx], gsmis[idx])
                dockedpose = oechem.OEMol()
                dockobj.DockMultiConformerMolecule(dockedpose, new_mol,1)
                ds = dockedpose.GetEnergy()
                ps = dockobj.ScoreLigand(new_mol)
                print("SCORE",ds ,ps )
                if return_docked_pose:
                    new_mol = oechem.OEMol(dockedpose)
                    new_mol2 = oechem.OEMol(dockedpose)

                if ps < 100:
                    scores.append(ps)
                    data.append((new_mol, new_mol2, gs, action))
                pscores.append(ps)
            except:
                continue
        order = np.argsort(scores)
        self.env.data['docking_scores'].append(scores)
        self.env.data['pose_scores'].append(pscores)

        data = [data[i] for i in order]
        return data

    def choose_action(self):

        self.env.openmm_simulation.get_pdb("test.pdb")
        pdb = oechem.OEMol()
        prot = oechem.OEMol()
        lig = oechem.OEMol()
        wat = oechem.OEGraphMol()
        other = oechem.OEGraphMol()
        ifs = oechem.oemolistream("test.pdb")
        oechem.OEReadMolecule(ifs, pdb)
        ifs.close()
        if not oechem.OESplitMolComplex(lig, prot, wat, other, pdb):
            print("crap")
            exit()

        self.env.action.update_mol_aligneer(lig)
        actions, gsmis = self.env.action.get_new_action_set()
        data = self.getscores(actions, gsmis, prot, num_returns=self.num_returns, return_docked_pose=self.return_docked_pose)
        not_worked = True
        idxs = list(range(len(data)))
        idx = idxs.pop(0)
        counter = 0
        while not_worked:
            try:
                new_mol, new_mol2, gs, action = data[idx]
                self.env.systemloader.reload_system(gs, new_mol, "test.pdb")
                self.env.openmm_simulation = self.env.config.openmmWrapper.get_obj(self.env.systemloader, ln=self.env.systemloader, stepSize=self.step_size * unit.femtoseconds)
                not_worked = False
            except Exception as e:
                print(e)
                if len(idxs) == 0:
                    print("mega fail")
                    exit()
                idx = idxs.pop(0)
        self.env.action.apply_action(new_mol2, action)

        return new_mol2, action
