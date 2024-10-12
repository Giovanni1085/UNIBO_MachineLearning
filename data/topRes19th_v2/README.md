# A Dataset for Toponym Resolution in Nineteenth-Century English Newspapers

## Description

We present a new dataset (version 2) for the task of toponym resolution in digitised historical newspapers in English. It consists of 455 annotated articles from newspapers based in four different locations in England (Manchester, Ashton-under-Lyne, Poole and Dorchester), published between 1780 and 1870. The articles have been manually annotated with mentions of places, which are linked---whenever possible---to their corresponding entry on Wikipedia. The dataset is published on the British Library shared research repository, and is especially of interest to researchers working on improving semantic access to historical newspaper content.

We share the 455 annotated files (one file per article) in the WebAnno TSV file format version 3.2, a CoNLL-based file format. The files are split into a train and test set. For each split, we additionally provide a TSV file with metadata at the article level. We also provide the original annotation guidelines.

## Directory structure

```bash=
topRes19th_v2/
├── README.md
├── train/
│   ├── annotated_tsv/
│   │   ├── 1218_Poole1860.tsv
│   │   ├── ...
│   │   └── 10877685_Dorchester1830.tsv
│   └── metadata.tsv
├── test/
│   ├── annotated_tsv/
│   │   ├── 9144_Poole1860.tsv
│   │   ├── ...
│   │   └── 10860796_Dorchester1860.tsv
│   └── metadata.tsv
└── original_guidelines.md
```

## Data description

### `[split]/annotated_tsv/*.tsv`

Each WebAnno TSV file in `annotated_tsv/` corresponds to an article. The file names (e.g. `1218_Poole1860.tsv`) consist of three elements: an internal Living with Machines identifier of the article (`1218`), the place of publication (`Poole`) and the decade of publication (`1860`). The WebAnno TSV format is a CoNLL-based file format, which has a header, is sentence-separated (by a blank line), and lists one token per line, with the different layers of annotations separated with tabs. See an example:
```
#FORMAT=WebAnno TSV 3.2
#T_SP=webanno.custom.Customentity|identifiier|value


#Text=THE POOLE AND SOUTH-WESTERN HERALD, THURSDAY, OCTOBER 20, 1864.
1-1	0-3	THE	_	_	
1-2	4-9	POOLE	_	_	
1-3	10-13	AND	_	_	
1-4	14-27	SOUTH-WESTERN	_	_	
1-5	28-34	HERALD	_	_	
1-6	34-35	,	_	_	
1-7	36-44	THURSDAY	_	_	
1-8	44-45	,	_	_	
1-9	46-53	OCTOBER	_	_	
1-10	54-56	20	_	_	
1-11	56-57	,	_	_	
1-12	58-62	1864	_	_	
1-13	62-63	.	_	_	

#Text=POOLE TOWN COUNCIL.
2-1	65-70	POOLE	https://en.wikipedia.org/wiki/Poole	LOC
2-2	71-75	TOWN	_	_	
2-3	76-83	COUNCIL	_	_	
2-4	83-84	.	_	_	
```

This example has two full sentences, preceded by `#Text=`, and split with one token per line. Now we look at one line in more detail:
```
2-1	65-70	POOLE	https://en.wikipedia.org/wiki/Poole	LOC	
```
The tab-separated elements are:
* `2-1`: the indices of the sentence in the document and the token in the sentence.
* `65-70`: start and end character positions of the token in the document.
* `POOLE`: the token.
* `https://en.wikipedia.org/wiki/Poole`: the Wikipedia url (if linked).
* `LOC`: the toponym class.

Toponyms are annotated with the following classes:
* `BUILDINGS`: names of buildings, such as the 'British Museum'.
* `STREET`: streets, roads, and other odonyms, such as 'Great Russell St'.
* `LOC`: any other real world places regardless of type or scale, such as 'Bloomsbury', 'London' or 'Great Britain'.
* `ALIEN`: extraterrestrial locations, such as 'Venus'.
* `FICTION`: fictional or mythical places, such as 'Hell'.
* `OTHER`: other types of entities with coordinates, such as events, like the 'Battle of Waterloo'.


### `metadata.tsv`

The `metadata.tsv` file links each annotated tsv file to its metadata. It consists of a header and one row per article, with the following fields:
* `fname`: name of the annotated file, without the extension (e.g. `1218_Poole1860`)
* `word_count`: number of words in the article.
* `ocr_quality_mean`: OCR quality mean, calculated as per-word OCR confidence scores as reported in the source metadata.
* `ocr_quality_sd`: OCR quality standard deviation.
* `issue_date`: date of publication of the article.
* `publication_code`: publication code (internal).
* `publication_title`: name of the newspaper publication.
* `decade`: decade of publication of the article.
* `place_publication`: place of publication.
* `annotation_batch`: each article is assigned to one annotation batch. All annotation batches are similarly-distributed in terms of place and decade of publication.

## License

The dataset is released under open license CC-BY-NC-SA, available at https://creativecommons.org/licenses/by-nc-sa/4.0/.

## Copyright notice

Newspaper data has been provided by Findmypast Limited from the British Newspaper Archive, a partnership between the British Library and Findmypast (https://www.britishnewspaperarchive.co.uk/).

## Funding statement

This work was supported by Living with Machines (AHRC grant AH/S01179X/1) and The Alan Turing Institute (EPSRC grant EP/N510129/1).

## Dataset creators

Mariona Coll Ardanuy (conceptualization, data curation, formal analysis, project management, writing), David Beavan (resources, software, writing), Kaspar Beelen (resources, data curation, writing), Kasra Hosseini (resources, software), Jon Lawrence (conceptualization, data curation, project management), Katherine McDonough (conceptualization, data curation, writing), Federico Nanni (validation, writing), Daniel van Strien (resources, software), Daniel C.S. Wilson (conceptualization, data curation, writing).

## Version changes

**Version 2:**

* Annotations (`annotated_tsv/*.tsv`):
    - The toponyms that were annotated as "LOCWiki" are now annotated as "LOC".
    - "UNKNOWN" has been removed from all data fields, instances of this class have been classified into the other classes (mostly "LOC").
* Metadata (`metadata.tsv`):
    - Column "publication_location" removed.
    - Column "annotation_decade" renamed to "decade".
    - Column "annotation_location" renamed to "place_publication".