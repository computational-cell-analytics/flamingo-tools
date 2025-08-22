# Segmentation and detection workflows

Implements workflows to segment SGNs or IHCs, and to detect ribbon synapses in slurm.

For SGN segmentation run:
- mean_std_SGN_template.sbatch
- apply_unet_SGN_template.sbatch
- segment_unet_SGN_template.sbatch

For IHC segmentation run:
- mean_std_IHC_template.sbatch
- apply_unet_IHC_template.sbatch
- segment_unet_IHC_template.sbatch

For ribbon synapse detection without associated IHC segmentation run
- detect_synapse_template.sbatch
For ribbon synapse detection with associated IHC segmentation run
- detect_synapse_marker_template.sbatch
