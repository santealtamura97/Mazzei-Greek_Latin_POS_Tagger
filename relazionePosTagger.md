## Pos Tagger per lingue morte: Latino e Greco
###  0. Introduzione
L'obiettivo dell'esercitazione è quello di implementare un Pos Tagger statistico per lingue morte (Latino e Greco) basato su **HMM** (Hidden Markov Model), il quale molto spesso viene utilizzato nella analisi delle sequenze.
L'operazione di Pos Tagging consiste in:

 - **Input**: una frase, vista come una sequenza di parole
 - **Output**: una lista di Pos Tag. Ogni tag è associato ad una parola della frase presa in input dall'algoritmo.

Questo problema, secondo lo standard del machine learning, è stato affrontato in 3 fasi, ma la definizione del modello (Modelling) è già nota agli addetti ai lavori:

 - **Modelling**: definizione del **modello matematico** del problema
 L'obiettivo è trovare la sequenza di tag $\hat{t}_1^n$ data la sequenza di parole osservabili $w_1^n$, che massimizza la distribuzione di probabilità $\text{P}$:
$$
\hat{t}_1^n = \underset{t_1^n}{\operatorname{argmax}}P(t_1^n,w_1^n)
$$
Per rendere operazionale questo calcolo è possibile sfruttare la regola di Bayes in modo tale da ottenere una equazione che avrà più probabilità da calcolare, ma **approssimabili**. In questo modo si ottiene una nuova equazione che approssima la precedente:
$$
\hat{t}_1^n = \underset{t_1^n}{\operatorname{argmax}}P(t_1^n,w_1^n) \approx  \prod_{i = 1}^{n} P(w_i \mid t_i)P(t_i \mid t_{i-1})
$$
Distinguiamo le due probabilità:
$$
P(w_i \mid t_i) \text{ e } P(t_i \mid t_{i-1})
$$
le quali sono rispettivamente le **probabilità di transizione** ed **emissione** utilizzate nella fase di Learning, che verranno trattate nel capitolo successivo insieme agli algortimi e strutture dati impiegati rispettivamente per ottenerli e memorizzarli

 - **Learning**:  apprendere i parametri da un dato corpus, che verranno utilizzati nella fase di Decoding
 - **Decoding**: implementazione dell'**algoritmo** di Pos Tagging

Infine, è stata fondamentale la valutazione dei risultati in modo da comprendere quale strategia di **smoothing** risulta essere più efficace e se le performance ottenute dell'algoritmo implementato superano, e in che modo, quelle del più comune algoritmo di **baseline**.  E' stato necessario, inoltre, confrontare i risultati ottenuti in base al linguaggio testato e analizzare gli **errori** più frequenti in modo tale da capire i limiti del sistema nei confronti delle lingue morte.

### 1. Implementazione del Learning

Le probabilità di transizione ed emissione vengono calcolate indipendentemente mediante i **train set** del greco e latino.

#### 1.1 Probabilità di transizione 
La probabilità di transizione rappresenta la probabilità di un tag dato il tag precedente. I **conteggi** vengono definiti rispettivamente dalla formula della probabilità:
$$
P(t_i,t_{i-1}) = \frac{C(t_{i-1},t_i)}{C(t_{i-1})}
$$

$$
C(t_{i-1}) = \text{numero di volte che il tag $t_{i-1}$ compare nel train set}
$$

$$
C(t_{i-1},t_i) = \text{numero di volte che il tag $t_{i-1}$ compare prima del tag $t_i$ all'interno del train set}
$$

![train1](/home/santealtamura/Immagini/train1.png)

Pertanto, nel file del train set, letto attraverso la libreria ***pyconll***, verrà considerata solo la lista dei tag relativi ad una sentence. 

`train = pyconll.load_from_file('la_llct-ud-train.conllu')`

Ogni parola della frase ha un attributo ***upos*** mediante il quale è possibile ricavare il Pos Tag. Di seguito la funzione per calcolare la probabilità di transizione tra due tag:

```
def compute_transition_probability(train,tag1,tag2):
   count_t1_before_t2 = 0
   count_t1 = 0
   for sentence in train:
       for i in range (len(sentence)):
           if sentence[i-1].upos == tag1 and sentence[i].upos == tag2 and i != 0:
               count_t1_before_t2 = count_t1_before_t2 + 1
           if sentence[i].upos == tag1:
               count_t1 = count_t1 + 1
   return count_t1_before_t2/count_t1
```

La funzione **compute_transition_probability** calcola la probabilità di transizione del tag2 dato il tag1, cioè conta quante volte il tag1 compare prima del tag2 e quante volte è presente il tag1 in tutte le **sentence** del train set. Restituisce la divisione tra i due conteggi.

#### 1.2 Matrice di transizione

Per la memorizzazione delle probabilità di transizione calcolate come descritto nel capitolo precedente è stata utilizzata una **matrice** etichettata sia sulle righe che sulle colonne con i Pos Tag per poter accedere alle probabilità in maniera semplice, specificando i due tag relativi ad essa. 

Le **matrici di transizione** cambiano in base al linguaggio utilizzato per il training perchè l'insieme dei Pos Tag relativi al train set del latino è diverso da quello relativo al greco antico. 

Infatti avremo una lista di ***possible_tags***:

- Pos Tags Latino:  [*'ADJ','ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'NOUN', 'NUM', 'PART', 'PRON','PROPN','PUNCT', 'SCONJ', 'VERB','X'*]
- Pos Tags Greco: [*'ADJ','ADP', 'ADV', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON','SCONJ', 'VERB', 'X', 'PUNCT'*]

Di seguito l'algoritmo utilizzato per popolare la matrice di transizione che sfrutta la funzione vista in precedenza per calcolare le probabilità di transizione relative a due tag. Il calcolo viene effettuato per ogni coppia di tag presente nella lista ***possible_tags***:

```
def compute_trasition_matrix(possible_tags,train):
    transition_matrix = np.zeros((len(possible_tags), len(possible_tags)), dtype='float32')
    for i,t1 in enumerate(possible_tags):
        for j,t2 in enumerate(possible_tags):
            transition_matrix[i][j] =  compute_transition_probability(train,t1,t2)
    return transition_matrix
```

Successivamente la matrice viene trasformata in un **DataFrame** per poter etichettare le righe e le colonne con i Pos Tag.

Un esempio di matrice di transizione relativa al train set per il Latino:

![matrice_transizione](/home/santealtamura/Immagini/matrice_transizione.png)

#### 1.3 Probabilità di transizione iniziali

In una struttura dati separata sono state memorizzate le probabilità di transizione iniziali, cioè le probabilità che da uno stato iniziale ***START*** si passi ad uno stato relativo ad uno dei Pos Tag presenti nella lista ***possible_tags***.

Ad esempio, per il Latino:

![initial_transirions](/home/santealtamura/Immagini/initial_transirions.png)

#### 1.4 Probabilità di emissione

La probabilità di emissione rappresenta quanto un certo tag sia una certa parola. Anche in questo caso i conteggi vengono definiti dalla formula per il calcolo della probabilità:
$$
P(w_i,t_{i}) = \frac{C(t_{i},w_i)}{C(t_{i})}
$$
$C(t_{i},w_i)$ = numero di volte che la parola $w_i$ è taggata con il tag $t_i$ nel train set

$C(t_{i})$ = numero di volte che il tag $t_i$ compare nel train set

#### 1.5 Dizionario di emissione

Per la memorizzazione delle probabilità di emissione è stato utilizzato un **dizionario** in quanto una matrice di emissione avrebbe avuto tante colonne quante sono le parole nel train set e non sarebbe una struttura dati ottimale.

Inoltre, il calcolo avviene a priori per tutte le parole del train set in modo tale da NON costruire un dizionario in maniera dinamica ogni volta che viene analizzata una sentence e rendere l'algoritmo di Decoding, presentato in un capitolo successivo, più veloce.

La seguente funzione restituisce due dizionari: il **dizionario di emissione** e quello delle parole con il loro relativo conteggio all'interno del train set. Questo secondo dizionario servirà nella fase di Decoding per verificare se una parola della sentence analizzata è sconosciuta o no.

```
def compute_emission_probabilities(train):
    word_tag_set = []
    tags_set = []
    words_set = []
    for sentence in train:
        for token in sentence:
            word_tag_set.append((token.form,token.upos))
            tags_set.append(token.upos)
            words_set.append(token.form)
            
    count_word_tag = dict(Counter(word_tag_set))
    count_tags = dict(Counter(tags_set))
    count_word = dict(Counter(words_set))
    
    emission_dict = dict()
    for key in count_word_tag:
        emission_dict[(key[0],key[1])] = count_word_tag[key]/count_tags[key[1]]

    return emission_dict,count_word
```

La funzione prende in input il train set (del Latino o del Greco) e restituisce **emission_dict**, un dizionario che avrà come chiavi delle coppie **(word,tag)** e come valori le **probabilità di emissione** relative alle coppie.

Un estratto del dizionario di emissione associato al train set del Latino:

```
{...,('Dei', 'PROPN'): 0.057208953357509064, ('nomine', 'NOUN'): 0.01572524239062274, ('regnante', 'VERB'): 0.015205799033494418, ('domno', 'NOUN'): 0.014205778785393855, ('nostro', 'DET'): 0.035255029840644804, ('Carulo', 'PROPN'): 0.004689258471926972, ('rege', 'NOUN'): 0.0035695335487916646, ('Francorum', 'NOUN'): 0.0027977425112150887,...}
```

Attraverso gli attributi ***form*** e ***upos*** è possibile accedere rispettivamente alla stringa del token della sentence in oggetto e al Pos Tag assegnato nel train set.

### 2. Implementazione del Decoding

La fase di Decoding permette di capire qual è l'algoritmo che permette di applicare le probabilità apprese nella fase di Learning per poter restituire la sequenza di tag ottimale per una determinata frase in input.

Dal momento che, la soluzione ottimale è la **sequenza che massimizza la probabilità** vista nel capitolo 0, per poterla ottenere si dovrebbero provare tutte le possibili combinazioni di tag associate ad una frase, calcolare la probabilità e scegliere quella massima. E' subito chiaro che c'è un problema! C'è un esplosione di casi e si avrebbe una **complessità esponenziale**.

La soluzione è quella di applicare la **dynamic programming** e passare da una complessità esponenziale ad una **polinomiale** attraverso l'**approssimazione Markoviana**.

Le due assunzioni fondamentali dell'HMM sono:

- La probabilità che una parola appaia dipende solo dal suo stesso tag ed è indipendente dalle parole vicine e dagli altri tag
- La probabilità di un tag dipende solo dal tag precedente anzichè che dall'intera sequenza di tag precedenti

L'idea è che la programmazione dinamica permette di non ripetere gli stessi calcoli più volte e consente di memorizzare i dati parziali, passando da una complessità temporale ad una spaziale.

L'**algoritmo di Viterbi** permette di fare tutto ciò, e nel capitolo seguente verrà spiegata la sua implementazione all'interno del progetto.

#### 2.1 Algoritmo di Viterbi

L'algoritmo di Viterbi implementato è una **variante** di quello tradizionale, ma con la stessa finalità. Si cercheranno di fare analogie con il secondo in modo tale da rendere più efficace la comprensione. 

Input dell'algoritmo:

- `sentence_tokens`: una lista di parole della frase analizzata
- `possible_tags`: lista dei possibili tag relativi dal train set utilizzato (Greco o Latino)
- `transition_matrix`
- `emission_probabilities`: dizionario delle probabilità di emissione
- `initial_transition_probabilities`
- `count_word`: dizionario delle parole con i relativi conteggi nel train set
- `smoothing_strategy`: strategia di smoothing utilizzata, relativa al train set
- `oneshot_words_tag_distribution`: distribuzione probabilistica dei tag relativi alle parole che compaiono una sola volta nel dev set

Per la realizzazione dell'algortimo non viene costruita una **matrice di Viterbi**, ma le colonne vengono simulate attraverso un **vettore** `p` dichiarato all'inizio del ciclo for il quale analizza una singola parola. 

L'algoritmo si compone quindi di due cicli, uno più esterno che scorre le parole della frase in input e uno interno che scorre su tutti i tag, ovvero gli **stati**.

Gli stati con le probabilità della matrice (i valori di Viterbi) vengono simulati e memorizzati nel vettore `p`, il quale si svuota ad ogni iterazione su una singola parola. 

Il vettore `states` è utilizzato per tenere traccia della sequenza di tag da restituire: ad ogni iterazione su una parola, viene scelto il tag dalla lista `possible_tags` in base all'indice del valore di Viterbi massimo nel vettore `p`, e questo viene inserito nella lista `states`. 

```
def viterbi_algorithm(sentence_tokens, possible_tags, transition_matrix, 
                      emission_probabilities, initial_transition_probabilities,
                      count_word,smoothing_strategy,
                      oneshot_words_tag_distribution):
    states = []   
    for key,word in enumerate(sentence_tokens):
        p = []
        for t,tag in enumerate(possible_tags):
            emission_p = 0
            if key == 0:
                trasition_p = initial_transition_probabilities.loc['START',tag]
            else:
                trasition_p = transition_matrix.loc[states[-1]][tag]
            try:
                count_word[word]
            except KeyError: #unknown_word
                emission_p = unknown_word_emission_p(smoothing_strategy, tag, possible_tags, oneshot_words_tag_distribution)         
            emission_p = emission_probabilities.get((word,tag),emission_p)
            if emission_p != 0 and trasition_p != 0:
                state_probability = math.log(emission_p) + math.log(trasition_p)
            else:
                state_probability = -sys.maxsize
            p.append(state_probability)
        pmax = max(p)
        state_max = possible_tags[p.index(pmax)]
        states.append(state_max)
    return states
```

All'interno del ciclo più annidato vengono ricavate le probabilità di **transizione** ed **emissione**. 

- Se l'indice della parola da analizzare è pari a 0 vuol dire che la probabilità di transizione è quella iniziale, memorizzata nella matrice `initial_transition_probabilities`, altrimenti viene ricavata prendendo in considerazione il tag per il quale il valore di viterbi della colonna precedente è massimo (ultimo elemento della lista `states`) e il tag corrente; questi due indici vengono usati per ricavare la probabiltà all'interno della matrice di transizione `transition_matrix` .
- La probabilità di emissione viene ricavata dal dizionario di emissione `emission_probabilities` se la parola considerata è conosciuta, altrimenti verrà generato un `KeyError` ; nel blocco di eccezione la funzione `unknown_word_emission_p` deciderà quale probabilità di emissione restituire in base alla **strategia di smoothing** utilizzata.
- La probabilità di uno stato sarà la somma dei logaritmi della due probabilità se entrambe sono diverse da 0, altrimenti avrà valore uguale a -infinito.

### 3. Strategie di Smoothing

Sono state utilizzate diverse strategie di **smoothing** per le parole sconosciute:

- $P(unk|NOUN) = 1$: se il tag preso in considerazione per la parola sconosciuta è ***NOUN*** allora la probabilità di emissione sarà uguale a 1, ovvero si decide con estrema certezza che quella parola sconosciuta è un **sostantivo**

- $P(unk|NOUN) = P(unk|VERB) = 0.5$: se il tag preso in considerazione per la parola sconosciuta e **NOUN** o **VERB** allora la probabilità di emissione sarà uguale a 0.5, ovvero la parola può essere un sostantivo o un verbo con la stessa probabilità

- $P(unk|t_i)= \frac{1}{|POS TAGs|}$: la proabilità di emissione per qualsiasi tag sarà la 1 diviso la cardinalità dell'insieme dei tag possibili, ovvero una parola sconosciuta può essere un **qualsiasi tag** della lista dei tag possibili con la stessa probabilità

- **Statistica PoS sul development set**: la probabilità di emissione viene calcolata sulla base di una distribuzione estratta da un determinato corpus. In pratica, la distribuzione è relativa ai tag associati alle parole che compaiono **una sola volta** nel dev-set.

  Il procedimento consiste nel collezionare tutte le parole che compaiono una sola volta (**one shot word**) e verificare quante di queste sono NOUN, VERB, ADJ, ecc... 

  Come tutte le distribuzioni, quella calcolata associerà ad ogni tag una probabilità e la somma di tutte le probabilità sarà 1. Nell'algoritmo `compute_oneshot_words_distributions`, che calcola e restituisce la distribuzione,  verranno considerati anche i tag non associati a nessuna one shot word presente nel dev-set; a questi verrà assegnata probabilità 0. 
  
  Ad esempio, la distribuzione di probabilità per i tag relativi alle one shot word del dev-set del Latino sarà:
  
  ```
  [('NOUN', 0.19340159271899887), ('PROPN', 0.38680318543799774), ('VERB', 0.229806598407281), ('ADJ', 0.080773606370876), ('CCONJ', 0.0011376564277588168), ('DET', 0.051194539249146756), ('NUM', 0.012514220705346985), ('ADP', 0.009101251422070534), ('ADV', 0.022753128555176336), ('PRON', 0.004550625711035267), ('AUX', 0.005688282138794084), ('SCONJ', 0.0011376564277588168), ('PART', 0.0011376564277588168), ('PUNCT', 0), ('X', 0)]
  ```
  
  E' possibile notare un'alta percentuale di nomi propri (PROPN).

### 4. Valutazione del Sistema e Analisi degli errori
