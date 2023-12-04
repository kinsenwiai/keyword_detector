# keyword_detector
Firstly run data_extractor.py on the dataset folder to get the texts from the files.
First stage extraction takes place first through pypdf and then by ocr from pytesseract.
"python data_extractor.py"

Now once you have the detected text, then the next goal is to translate everything to englisj
using a translation model and also the sentiment verdict was added with a pre-trained
sentiment analysis model.
"python sent_analysis.py" inside the sent_analysis folder

Next once you have everything added, finally run the keyword matching algorithm,
that executes the keyword_matching.py.
"python keyword_matching.py"
