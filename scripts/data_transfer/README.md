# Data Transfer Moser

## Transfer via smbclient

Current approach to the data transfer:
- Log in to SCC login node:
  $ ssh -i ~/.ssh/id_rsa_scc pape41@transfer-mdc.hpc.gwdg.de
- Go to "/scratch1/projects/cca/data/moser"
- Create subfolder <NAME> for cochlea to be copied 
- Log in via $ smbclient \\\\wfs-medizin.top.gwdg.de\\ukon-all\$\\ukon100 -U GWDG\\pape41"
- Go to the folder with the cochlea to copy (cd works)
- Copy the folder via:
    - recurse ON
    - prompt OFF
    - mget *
- Copy this to HLRN by logging into it and running
  $ rsync  pape41:/scratch1/projects/cca/data/moser/<NAME>
  $ rsync -e "ssh -i ~/.ssh/id_rsa_hlrn" -avz pape41@login-mdc.hpc.gwdg.de:/scratch1/projects/cca/data/moser/<NAME> /mnt/lustre-grete/usr/u12086/moser/lightsheet/<NAME>
- Remove on SCC

## Next files

- UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\2024\M171_2R_converted_n5
- UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\2024\155_1L_converted_n5
- UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\2024\MLR151_2R_converted_n5
- UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\2024\G11_1L_converted_n5

## Improvements

Try to automate via https://github.com/jborean93/smbprotocol see `sync_smb.py` for ChatGPT's inital version.

## Transfer Back

For transfering back MoBIE results.
...

# Data Transfer Huisken

See "Transfer via smbclient" above:
```
smbclient \\\\wfs-biologie-spezial.top.gwdg.de\\UBM1-all\$\\ -U GWDG\\pape41
```
