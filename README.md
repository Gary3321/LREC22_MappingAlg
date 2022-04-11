# LREC22_MappingAlg
This repo is the code release of LREC 2022 conference paper "Lexical Resource Mapping via Translations".

## Tested Environment

- Python: 3.8.8
- Java: OpenJDK 1.8.0_201

## Data
Extract the following data into the data format of sample files in "data/".

- [BabelNet 4.0](https://babelnet.org/guide) - We used the java BabelNet API version 4.0.1
- [CLICS](https://clics.clld.org/) - We followed this guide (https://github.com/clics/clics2/tree/master/cookbook) to obtain the CLICS data.
- [OmegaWiki](http://omegawiki.org/)

The gold data used in our experiments can be found below:

- [WordNet-CLICS gold mapping][https://github.com/concepticon/concepticon-data/blob/master/concepticondata/concept_set_meta/wordnet.tsv]
- [WordNet-OmegaWiki gold mapping][http://lcl.uniroma1.it/semalign/]

## Run the mapping algorithms

Map concepts between WordNet and CLICS

```
cd Code
python Map_Clics_WN.py

```

Map concepts between WordNet and OmegaWiki
```
cd Code
python Map_OW_WN.py

```
