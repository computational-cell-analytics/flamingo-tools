# Data Transfer Moser

## Transfer via smbclient

Current approach to the data transfer:
- Log in to SCC login node:
  $ 
- Go to `/scratch1/projects/cca/data/moser`
- Create subfolder <NAME> for cochlea to be copied 
- Log in via 
```
$ smbclient \\\\wfs-medizin.top.gwdg.de\\ukon-all\$\\ukon100 -U GWDG\\pape41"
```
- Go to the folder with the cochlea to copy (cd works)
- Copy the folder via:
    - recurse ON
    - prompt OFF
    - mget *
- Copy this to HLRN by logging into it and running
```
  $ rsync -e "ssh -i ~/.ssh/id_rsa_hlrn" -avz pape41@login-mdc.hpc.gwdg.de:/scratch1/projects/cca/data/moser/<NAME> /mnt/lustre-emmy-hdd/projects/nim00007/data/moser/lightsheet/volumes/<NAME>
```
- Remove on SCC

## Next files

- UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\2024\M171_2R_converted_n5
    - unclear what the converted data is
- UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\2024\155_1L_converted_n5\BDVexport.n5
    - Copied to SCC, rsync in progress.
- UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\2024\MLR151_2R_converted_n5
    - unclear what the converted data is
- UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\2024\G11_1L_converted_n5
    - unclear what the converted data is

## Improvements

Try to automate via https://github.com/jborean93/smbprotocol see `sync_smb.py` for ChatGPT's inital version.
Connection not possible from HLRN.

## Transfer Back

For transfering back MoBIE results.
...
