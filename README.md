# Common Voice based language recognition deep learning project
This repository is for a university deep learning course final project - Common Voice based language recognition.

## Dataset: [Common Voice](https://mozilladatacollective.com/organization/cmfh0j9o10006ns07jq45h7xk)

We chose 3 languages (Polish, Dutch, Portuguese) from 3 different language families (Slavic, Germanic, Romanic) to minimize issues from the very start. First problem arose with picking out similarly sized datasets, we're either blind or there's no way to filter through Common Voice bank.



### What to do to load the datasets?

You'll need to register at Mozilla Data Collective and create your MDC_API_KEY (put it in .env file). Then, run src/dataset.py.


## Project structure
```
common-voice-language-recognition/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── models/
    ├── best_model.keras
    └── language_recognition_model.keras
└── src/
    ├── train.py
    ├── train_on_own.py
    ├── dataset.py
    └── data/
        ├── labels_processed.csv
        ├── labels_augmented.csv
        ├── raw/ <-- will only appear locally after running dataset.py
        ├── processed/ <-- will only appear locally after running dataset.py
        └── processed_augmented/ <-- will only appear locally after running dataset.py
```










## Little 🧁 treat at the very end

We decided to test the model on some very odd (read: 'self-made') data!  Weronika Kłujszo - [@Nixx-K](https://github.com/Nixx-K) recorded samples for Polish and Portuguese of quality that definitely left some doubts, since she's only fluent in Polish, English and Spanish (but these datasets were just wayyy too big) and we checked the results.

Each language got 10 3-second samples with noise (some more, some less) - not artifically augmented, just recorded in everyday environment instead of studio. Portuguese got 100% accuracy, while Polish got... 0%. It got classified as Portuguese 60% of the time, while as Dutch - 40% of the time. We suspected it would mostly be classified as Portuguese, so the results were quite surprising. Most of the tie, though, the model was very sure it would be Portuguese (0.9 on average), but only semi-sure it would be Dutch (0.59 on average).
