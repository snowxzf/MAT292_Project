#!/bin/bash

python3 << 'EOF'
#!/usr/bin/env python3
"""
Extract tumor volumes from all HD-GLIO segmentations
"""

import os
from pathlib import Path
import pandas as pd

basepath = Path("/mnt/c/Users/jessi/Desktop/MAT_Project/Imaging")

results = []

for patient_dir in sorted(basepath.glob("Patient-*")):
    patient_id = patient_dir.name
    
    for week_dir in sorted(patient_dir.glob("week-*")):
        week_id = week_dir.name
        
        # Check for volumes.txt
        vol_file = week_dir / "volumes.txt"
        
        if vol_file.exists():
            # Parse volumes.txt
            volumes = {}
            with open(vol_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':')
                        volumes[key.strip()] = float(value.strip())
            
            results.append({
                'Patient': patient_id,
                'Week': week_id,
                'Edema_mm3': volumes.get('volume_non_enhancing_T2_FLAIR_signal_abnormality_mm3', 0),
                'Enhancing_mm3': volumes.get('volume_contrast_enhancing_tumor_mm3', 0),
                'Total_mm3': volumes.get('volume_non_enhancing_T2_FLAIR_signal_abnormality_mm3', 0) + 
                            volumes.get('volume_contrast_enhancing_tumor_mm3', 0)
            })
            print(f"✓ {patient_id}/{week_id}")
        else:
            print(f"⚠ {patient_id}/{week_id} - No volumes.txt")

# Save to CSV
df = pd.DataFrame(results)
df.to_csv('all_tumor_volumes_hdglio.csv', index=False)

print(f"\n✓ Saved {len(results)} timepoints to all_tumor_volumes_hdglio.csv")
print(f"\nSummary:")
print(f"  Patients: {df['Patient'].nunique()}")
print(f"  Total timepoints: {len(df)}")
print(f"  Avg tumor volume: {df['Total_mm3'].mean():.0f} mm³")
EOF

echo "Done"
