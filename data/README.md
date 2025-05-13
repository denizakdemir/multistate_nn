# Multistate Dataset Collection

This directory contains datasets commonly used for multistate modeling:

- **cav.csv**: Heart transplant coronary allograft vasculopathy data
- **bladder.csv**: Bladder cancer recurrence data
- **pbc.csv**: Primary biliary cirrhosis data
- **aidssi.csv**: AIDS and SI switching data

## Dataset Details

### ebmt
Bone marrow transplant data with states:
- Transplant
- Recovery (platelet recovery)
- Adverse event (acute graft-versus-host disease)
- Relapse
- Death

### cav
Heart transplant data with coronary allograft vasculopathy (CAV) states:
- No CAV
- Mild/moderate CAV
- Severe CAV
- Death

### bladder
Bladder cancer recurrence data with multiple recurrence events.

### pbc
Primary biliary cirrhosis data with disease progression states.

### aidssi
AIDS and SI switching data with:
- HIV infection
- SI virus appearance
- AIDS diagnosis
- Death

## Usage with MultiStateNN

These datasets need preprocessing before use with MultiStateNN. See the examples directory for notebooks demonstrating proper data preparation.
