#!/bin/bash

basepath="/mnt/c/Users/jessi/Desktop/MAT_Project/Imaging"

echo "Creating volumes.txt for all segmentations..."
echo ""

total=0
created=0
skipped=0
missing=0

for patient_dir in "$basepath"/Patient-*; do
    if [ ! -d "$patient_dir" ]; then continue; fi
    
    patient_name=$(basename "$patient_dir")
    
    for week_dir in "$patient_dir"/week-*; do
        if [ ! -d "$week_dir" ]; then continue; fi
        
        week_name=$(basename "$week_dir")
        ((total++))
        
        # Skip if volumes.txt already exists
        if [ -f "$week_dir/volumes.txt" ]; then
            ((skipped++))
            continue
        fi
        
        # Check if segmentation exists
        if [ ! -f "$week_dir/segmentation.nii.gz" ]; then
            echo "$patient_name/$week_name - No segmentation.nii.gz"
            ((missing++))
            continue
        fi
        
        # Create volumes.txt using Python
        python3 << EOF
import nibabel as nib
import numpy as np

seg_path = "$week_dir/segmentation.nii.gz"
seg = nib.load(seg_path)
data = seg.get_fdata()
voxel_vol = np.prod(seg.header.get_zooms())

vol_edema = (data == 1).sum() * voxel_vol
vol_enhancing = (data == 2).sum() * voxel_vol

output_path = "$week_dir/volumes.txt"
with open(output_path, 'w') as f:
    f.write(f"volume_non_enhancing_T2_FLAIR_signal_abnormality_mm3: {vol_edema:.2f}\n")
    f.write(f"volume_contrast_enhancing_tumor_mm3: {vol_enhancing:.2f}\n")
EOF
        
        echo "✓ $patient_name/$week_name"
        ((created++))
    done
done

echo ""
echo "========================================="
echo "Summary:"
echo "  Total weeks: $total"
echo "  Created volumes.txt: $created"
echo "  Already existed: $skipped"
echo "  Missing segmentation: $missing"
echo "========================================="

