# LMML - Stratul slab (Weak Layer)

Rezolvarea provocării **Stratul slab** din cadrul LMML (Liga de Machine Learning).

## Descrierea Sarcinii

În profilul mantalei de zăpadă, apariția stratului slab și cedarea pantei sunt determinate printr-un lanț conjunctive de condiții fizice. Sarcina presupune:
- Antrenarea unui clasificator arbore de decizie (`DecisionTreeClassifier`) pe setul de date primit.
- Recuperarea cuvântului ascuns codificat în ordinea întrebărilor / deciziilor din structura arborelui.
- Fiecare nod de decizie de-a lungul lanțului către cedarea pantei (`slab_released == 1`) utilizează o variabilă a cărei primă literă compune cuvântul ascuns.
- Format flag: `SIGMOID_<WORD>`.

## Rulare

```bash
pip install -r requirements.txt
python main.py
```

La rulare, scriptul citește `test-input.csv` din directorul curent și scrie rezultatul în `test-output.txt`.
