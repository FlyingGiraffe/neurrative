# Narrative Tension Field: A Geometric View of Story Structure

**[Project document](https://docs.google.com/document/d/14Wmcna-hrG1aNa0YlczN_qigWfSBBZtYJDHj2jzXlR0/edit?usp=sharing)**
|
**[SGI website](https://sgi.mit.edu/)**
|
**[Justin's geometry course](https://groups.csail.mit.edu/gdpgroup/68410_spring_2023.html)**

### File structure

```
books/
  alice_wonderland/
    raw.txt
    processed.json
    paragraph_scores.json
    embeddings/
      bge-m3/
      qwen3-embedding/
      e5-large-v2/
```

### Preprocessing

Run `preprocess/preprocess.py` to split the chapters and paragraphs in the book. <br>
Run `preprocess/get_embeddings.py` to obtain the text embeddings for each paragraph. <br>
Run `preprocess/get_annotations.py` to get the per-paragraph scores annotated by Qwen2.5-7B-Instruct.

**Download the preprocessed files:** **[link](https://drive.google.com/drive/folders/1Ut6MJmPx0LbXRXq15MGAbfBs9oUBT0Zj?usp=sharing)**.

### Implementation

Implement geometric algorithms in `geometry/`. <br>
Implement visualization scripts in `visualization/`.