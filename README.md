# Tarantino Script Generator
Code and data to build a character-level, bidirectional LSTM trained to generate text like Quentin Tarantino's screenplays. See more info in this [short post](http://perzan.io/projects/script-generator/).

#### There are 4 directories contained within this repo:
    **clean_text**: Landing spot for processed text files
    **raw_text**:   Raw text files ripped from the vector pdfs of each screenplay
    **outout**:     Output directory for saved model weights
    **src**:        Collection of user-defined functions for building the model
    
#### Running the model:
The Jupyter notebook **TarantinoMovieGeneration.ipynb** walks through data processing, building the model, 
training the model, and generating text.

#### Requirements:
    python==3.7.4
    keras==2.2.4
    tensorflow==1.14.0
    numpy==1.17.1
    sklearn==0.21.3
