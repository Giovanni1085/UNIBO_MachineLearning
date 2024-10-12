# Annotation guidelines

**Note:** These are the original annotation guidelines. The annotations in the current version of the dataset have since been refined, main changes are described in the accompanying README file (under `Version changes`).

## Toponym resolution annotations

This document contains the annotation guidelines for marking up and georeferencing locations mentioned in historical newspaper articles. The task of the annotator is to recognize each location mentioned in the text and map it to the URL of the Wikipedia article that refers to it.

### The task

Place names are often highly ambiguous. There are, for instance, more than 20 different places named Paris all over the world, as well as different instances of records relating to Paris, France. The task of toponym resolution can be similar to word sense disambiguation: in most scenarios the most commonly used sense (or place) is in most cases the correct sense (or place). However, our data is mostly composed of historical local and regional newspapers, and the world view that is represented in these texts is skewed towards the knowledge expected of their intended average, regional reader. It is therefore important that annotators take into account the date and place(s) of newspaper publication/circulation during the annotation process.

### What to annotate

Location: any named entity of a location that is static and can be defined according to a pair of static world coordinates (including metonyms, as in 'France signed the deal.'). If there is an OCR error, we will annotate the location if we can recognise it because of context clues in the word itself or in the surrounding text (for example, we would link "iHancfjrcter" to https://en.wikipedia.org/wiki/Manchester). We will not perform any additional post-correction of the OCRed text.

### How to annotate

The annotator should map each location found in the text with the URL of the Wikipedia article that refers to it.

To do so:
* Make sure you have selected the Layer `Custom entity` (if you don't see it, make sure you are in a 'Toponym resolution' project).
* Select with the mouse the span of text you want to annotate (e.g. 'West Laviogton') and select `LOCWiki` from the dropdown menu.

In this task, the custom entity `LOCWiki` refers to a real world place regardless of scale (region, city, neighborhood) with the exception of the additional, separate categories listed below:
  * `BUILDING`: Names of buildings (e.g. schools, hospitals, factories, palaces, etc.). Optional link to Wikipedia article if it exists.
  * `STREET`: Streets, squares, etc. Optional link to Wikipedia article if it exists.
  * `ALIEN`: Extraterrestrial locations (e.g. the moon). Optional link to Wikipedia article if it exists.
  * `OTHER`: Others, as in famous trees (https://en.wikipedia.org/wiki/Lone_Cypress) or battlefields (https://en.wikipedia.org/wiki/Battle_of_Waterloo). Optional link to Wikipedia article if it exists.
  * `UNKNOWN`: If the location has no Wikipedia entry OR if you cannot determine what place it is, but are confident that it is a place. No link to Wikipedia.
  * `FICTION`: If it is a fictional/mythical place (e.g. Lilliput). Optional link to Wikipedia article if it exists.

* How to annotate with Wikipedia links:
  * Go to Wikipedia (English version).
  * Find the correct article corresponding to the place mentioned in the text (e.g. `https://en.wikipedia.org/wiki/West_Lavington,_Wiltshire`).
  * Copy the full URL and paste it to the identifier box.
* To delete an annotation, click on it and click on `Delete` in the Annotation box.

The article title will give you an indication of the place of publication of the article, to help you disambiguate the toponyms in the article (e.g. `10713959_Dorchester1820.txt` is an article published in Dorchester, Dorset, in the 1820s---the date refers to the decade, not the year, of publication).

Some annotation considerations:
* Choose 'historical county' record over 'ceremonial county' for county place names.
* Do not include places that are not referred to by proper names (e.g. 'the park').
* Always favour a geo-coded link even if it is less perfect.
  > For example: Bengal---a province of British Colonial India---has a wiki page but it is not geo-coded because it is an historic term for places now in India (West Bengal) and Bangladesh. The latter has been linked since it represents the bulk of British Bengal and is geo-coded.
* Do not geocode the place if it's part of a person's title ("the Earl of Warwick").
* Company stocks and shares names after places - e.g. Westminster Bank, Devon Great Consols (mine) should NOT be linked as it is a commercial credit note linked to a trading entity. It isn't a place as such.