from astropy.io import ascii
import glob
import os
import sys
import copy

"""

Script for validating kilonova model files for Dave's GW190425 paper

"""

def is_number(val):
    try:
        val = float(val)
        return(True)
    except ValueError:
        return(False)
    return(False)

parent_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
print(parent_dir)

delete_bad_file = True

for kappa in [0.5, 1.0, 3.0, 3.65, 10.0]:

    kstr = '%.4f'%kappa

    glob_str = os.path.join(parent_dir, 'output', 'tables', f'mosfit_kn_{kstr}',
        '*.dat')
    print(glob_str)

    files = glob.glob(glob_str)

    for file in files:

        print(f'Checking {file}')

        table = ascii.read(file)

        if 'comment' not in table.meta.keys():
            if delete_bad_file:
                print(f'WARNING: comment not in file {file}')
                os.remove(file)
                continue
            else:
                raise Exception(f'Comment not in {file}')

        comment = list(table.meta['comment'])
        comment_data = {}

        # Fixing error in generation of tables
        if 'kappa' in comment[0] and 'type' in comment[0]:
            item_data = comment[0].split('type')
            kappa_data = item_data[0]
            type_data = item_data[1]
            type_data = 'type'+type_data
            kappa_data = kappa_data.strip()
            type_data = type_data.strip()
            new_comment = [kappa_data,type_data]
            new_comment.extend(comment[1:])
            
            table.meta['comment'] = new_comment
            table.write(file, overwrite=True, format='ascii.ecsv')
            comment = copy.copy(new_comment)

        for item in comment:
            key, val = item.split('=')
            key = key.strip()
            val = val.strip()
            if is_number(val):
                val = float(val)

            comment_data[key]=val

        cont = False
        for key in ['mass','velocity','type','kappa','theta_obs','tfloor']:
            if key not in comment_data.keys():
                if delete_bad_file:
                    print(f'WARNING: {key} not in metadata for file {file}')
                    os.remove(file)
                    cont = True
                    break
                else:
                    raise Exception(f'{key} not in metadata for {file}')
        if cont: continue

        for key in ['time','PS1_g','PS1_i','PS1_r','PS1_w','PS1_y','PS1_z',
            'swift_uvot_B','swift_uvot_U','swift_uvot_uvm2',
            'swift_uvot_uvw1','swift_uvot_uvw2','swift_uvot_V','johnson_B',
            'johnson_I','johnson_R','johnson_U','johnson_V','sdss_g',
            'sdss_i','sdss_r','sdss_u','sdss_z','ATLAS_o','ATLAS_c',
            'ukirt_H','ukirt_J','ukirt_K','Clear']:
            if key not in table.keys():
                if delete_bad_file:
                    print(f'WARNING: {key} not in {file}')
                    os.remove(file)
                    break
                else:
                    raise Exception(f'{key} not in {file}')
