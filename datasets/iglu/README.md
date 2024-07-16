### IGLU Augmentations Dataset

This dataset contains various files with different augmentation techniques applied to training data, specifically in the context of coordinates and primitives as subtasks.

#### Files Included:

1. **coords.csv** - The original training dataset with coordinates as subtasks.
2. **coordsAB.csv** - The original coordinates training dataset with ChatGPT and color augmentations applied.
3. **prims.csv** - The original training dataset with primitives as subtasks.
4. **primsAB.csv** - The original primitives training dataset with ChatGPT and color augmentations applied.

#### Columns in Each File:

- **description**: Instructions for the AI agent in natural language.
- **subtasks**: A deterministic list of subtasks derived from each instruction.

#### Subtask Formats:

The following table provides examples of the subtask formats for the "coords" and "prims" datasets in the IGLU environment. The instruction used for the example is: *"Architect, place 5 red blocks in a row, one row north of center."*

| **Format** | **Description** | **Example** |
|------------|-----------------|-------------|
| **coords** | (x, y, z, colorID):<br> - **x, y, z**: coordinates<br> - **colorID**: block color id | (0, 5, 5, 3), (0, 6, 5, 3), (0, 7, 5, 3), (0, 8, 5, 3), (0, 9, 5, 3) |
| **prims**  | (start), (size), rotation, color:<br> - **(start)**: initial block (x, y, z)<br> - **(size)**: dimensions (x, y, z)<br> - **rotation**: alignment<br> - **color**: block color name | (0, 5, 5), (1, 1, 5), eastsky, red |

This table helps in understanding the specific formats and how instructions are converted into subtasks for the AI agent within the IGLU environment.