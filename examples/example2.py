from datetime import datetime
from rlmm.environment.openmmEnv import OpenMMEnv
from rlmm.utils.config import Config
from rlmm.rl.Expert import  ExpertPolicy, RandomPolicy
import pickle
import numpy as np
import time
import os
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.DataStructs import cDataStructs
from plip.structure.preparation import PDBComplex



def setup_temp_files(config):
    try:
        os.mkdir(config.configs['tempdir'])
    except FileExistsError:
        pass
    if config.configs['tempdir'][-1] != '/':
        config.configs['tempdir'] = config.configs['tempdir'] + "/"
    config.configs['tempdir'] = config.configs['tempdir'] + "{}/".format( datetime.now().strftime("rlmm_%d_%m_%YT%H%M%S"))
    try:
        os.mkdir(config.configs['tempdir'])
    except FileExistsError:
        print("Somehow the directory already exists... exiting")
        exit()

    for k ,v in config.configs.items():
        if k in ['actions', 'systemloader', 'openmmWrapper', 'obsmethods']:
            for k_, v_ in config.configs.items():
                if k_ != k:
                    v.update(k_, v_)

def test_load_test_system(dir):
    import logging
    import warnings
    import shutil
    from openeye import oechem
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Warning)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.getLogger('openforcefield').setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")

    config = Config.load_yaml('examples/example2_config.yaml')
    setup_temp_files(config)
    shutil.copy('rlmm/tests/test_config.yaml', config.configs['tempdir'] + "config.yaml")
    env = OpenMMEnv(OpenMMEnv.Config(config.configs))
    policy = ExpertPolicy(env,num_returns=-1,sort='iscores', orig_pdb=config.configs['systemloader'].pdb_file_name)

    obs = env.reset()
    
    energies = []
    for i in range(100):
        choice = policy.choose_action(obs)
        print("***********Action taken: ", choice[1])
        print("first other item in tuple:", choice[0])
        #print("OBS:", obs) --> Obs is pdb file content
        obs, reward, done, data = env.step(choice)
        energies.append(data['energies'])
        
        # Convert action to fingerprint
        fp_mol = Chem.MolFromSmiles(choice[1])
        fp = AllChem.GetMorganFingerprintAsBitVect(fp_mol, 2, nBits=1024)
        action_vector = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, action_vector)
        print("CONVERTED ACTION TO FINGERPRINT!!:", action_vector)

        # Convert pdb format to vector
        # this is probably not super useful, as i dont fully understand it...
        #I stole this idea because it sounded like a useful way to vectorize the protein-ligand info:
        """
        Protein-Ligand interaction fingerprints (PLIFs) are becoming more widely 
        used to compare small molecules in the context of a protein target. 
        A fingerprint is a bit vector that is used to represent a small molecule. 
        Fingerprints of molecules can then be compared to determine the similarity between two molecules. 
        Rather than using the features of the ligand to build the fingerprint, a PLIF is based on the 
        interactions between the protein and the small molecule. The conventional method of building a 
        PLIF is that each bit of the bit vector represents a residue in the binding pocket of the protein. 
        The bit is set to 1 if the molecule forms an interaction with the residue,
        whereas it is set to 0 if it does not."""
        #https://www.blopig.com/blog/2016/11/how-to-calculate-plifs-using-rdkit-and-plip/

        my_mol = PDBComplex()
        my_mol.load_pdb(obs, as_string=True) # Load the PDB file into PLIP class
        my_mol.analyze()
        print("WANTING TO DO PLIF STUFF...")
        #for k, v in my_mol.interaction_sets.items():
            #print(k, v.all_itypes)
            #print()

        
        binding_sites=my_mol.interaction_sets.keys()
        interactions = my_mol.interaction_sets.values()
        interacting_res = []
        plif_vec = []
        for inter in interactions:
            for ii in inter.all_itypes:
                if len(ii) ==0:
                    plif_vec.append(0)
                # Exclude empty chains (coming from ligand as a target, from metal complexes)
                elif ii.restype !='HOH' and ii.reschain not in [' ', None]:
                    plif_vec.append(1)
                    interacting_res.append(''.join([str(ii.resnr), str(ii.reschain)]))
                else:
                    plif_vec.append(0)


        #all_res = [list(set([''.join([str(i.resnr), str(i.reschain)]) for i in interaction.all_itypes])) for interaction in interactions]
        #interacting_res = [list(set([''.join([str(i.resnr), str(i.reschain)]) for i in interactions if i.restype not in ['LIG', 'HOH']]))
        
        # So basically this seems really reductive, but I am going to make a vector that 
        # is as long as all the residues enumerated within the PLInteraction object's .all_itypes
        # property. Then if there is no interaction, the bit should stay 0 otherwise it will get 
        # turned to 1. I am also going to save off text about the actual interactions
        # to look at later. This likely was a waste of time, lol...


        """plif_vec = DataStructs.ExplicitBitVect(len(all_res), False)
        for index, res in enumerate(all_res):
            if res in interacting_res:
                plif_vec.SetBit(index)
            else:
                continue"""

        print("CONVERTED PDB TO VECTOR REPRESENTING PLIF!!:", plif_vec)
        print("INTERACTING_RES:", interacting_res)
        q_learning_vals = {
            "Observations": obs,
            "Reward": reward,
            "Action": action_vector,
            "Environment": plif_vec
            }

        with open('{}/Q_logs_step_{}.pickle'.format(dir,i), 'wb') as handle:
            pickle.dump(q_learning_vals, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open( config.configs['tempdir'] + "rundata.pkl", 'wb') as f:
            pickle.dump(env.data, f)


if __name__ == '__main__':
    timestr = time.strftime("%Y-%m-%d__%H_%M_%S")
    os.mkdir(timestr)
    test_load_test_system(timestr)
