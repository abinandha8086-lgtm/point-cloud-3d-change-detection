# 3D Point Cloud Change Detection (3DCDNet)

This repository implements the 3DCDNet architecture for detecting structural changes between 3D point cloud datasets. It includes a synthetic **Cube Example** to demonstrate the model's ability to identify point-level differences.

---

# The Cube Example
The "Cube Example" consists of two synthetic point clouds:
1. **Initial State:** A perfectly formed 3D cube.
2. **Changed State:** The same cube with a subset of points shifted or removed.
The model processes these inputs and generates a prediction that highlights exactly where the geometry was altered.


# commands to set up the project :

1. Clone the Repository
Open your terminal and run:

git clone [https://github.com/abinandha8086-lgtm/point-cloud-3d-change-detection.git](https://github.com/abinandha8086-lgtm/point-cloud-3d-change-detection.git)
cd point-cloud-3d-change-detection

2. Create and Activate Environment
It is recommended to use a virtual environment to avoid library conflicts

3. Install Requirements
Install the necessary deep learning and visualization libraries:

pip install --upgrade pip
pip install -r requirements.txt

4. Run the Visualization
To see the cube change detection in action, run the visualization script. This will open a 3D window showing the processed point cloud results:

   bash

   python3 viz_change.py 


# Example 
[Screencast from 2025-12-17 16-35-14.webm](https://github.com/user-attachments/assets/d52c6240-8d56-45b6-8923-bd0b7d972424)

