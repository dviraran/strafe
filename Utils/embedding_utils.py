from gensim.models.word2vec import LineSentence, Word2Vec, KeyedVectors
import random
import datetime


def train_embedding(featureSet, feature_matrix_3d_transpose, window_days, person_ixs, time_ixs, good_time_ixs,
                    embedding_dim, embedding_filename):
    data_filename = "wtv_data_window_{}d.txt".format(window_days)
    f = open(data_filename, "a")
    c = list(zip(person_ixs, time_ixs))
    c_unique = sorted(list(set(c)))
    until = None
    s = []
    last_p = None
    for i, (p_ix, t_ix) in enumerate(c_unique):
        if p_ix != last_p:
            until = None
            last_p = p_ix
        t = featureSet.time_map[good_time_ixs[t_ix]]
        # t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
        t = datetime.datetime.strptime(t, '%Y-%m-%d')
        if until is None or t > until:
            if s:
                for _ in range(5):
                    f.write(' '.join(str(w) for w in random.sample(s, len(s))))
                    f.write('\n')
            s = []
            until = t + datetime.timedelta(days=window_days)
        s += feature_matrix_3d_transpose[p_ix, t_ix, :].coords[0].tolist()
        if i % 500000 == 0:
            print('Wrote {} of {} to file'.format(i, len(c_unique)))
    f.close()

    sentences = LineSentence(data_filename)
    model = Word2Vec(min_count=5, vector_size=embedding_dim, workers=10)
    model.build_vocab(sentences)
    n_epochs = 20
    model.train(sentences, total_examples=model.corpus_count, epochs=n_epochs)

    # model.save("wtv_data_window_{}d_20epochs_model".format(window_days))
    # model.wv.save("wtv_data_window_{}d_20epochs_wv".format(window_days))
    # return "wtv_data_window_{}d_20epochs_wv".format(window_days)
    model.save(f"{embedding_filename}_model")
    model.wv.save(embedding_filename)
    return embedding_filename


def plot_embedding(model_filename):
    from sklearn.decomposition import IncrementalPCA  # inital reduction
    from sklearn.manifold import TSNE  # final reduction
    import numpy as np  # array handling
    import matplotlib.pyplot as plt
    import random
    model = Word2Vec.load(model_filename)
    num_dimensions = 2
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)
    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    random.seed(0)
    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
    plt.savefig(f"{model_filename}.png")


def closest_embedding(model_filename, featureSet, all_concept_map_reverse):
    all_concept_map = {v: k for k, v in all_concept_map_reverse.items()}

    from sklearn.decomposition import IncrementalPCA  # inital reduction
    from sklearn.manifold import TSNE  # final reduction
    import numpy as np  # array handling
    model = Word2Vec.load(model_filename)
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)
    n = len(labels)
    #distances_table = np.zeros((n, n))
    #ltn = {l:i for l,i in zip (labels, list(range(n)))}
    #ntl = {i:l for l,i in zip (labels, list(range(n)))}
    #for v1, l1 in zip(vectors, labels):
    #    for v2, l2 in zip(vectors, labels):
    #        if l1 == l2: distances_table[ltn[l1]][ltn[l2]] = 10*6
    #        distances_table[ltn[l1]][ltn[l2]] = np.linalg.norm(v1-v2)
    #closests = distances_table.argmin(axis=1)
    #for label in labels:
    #    print(f"The closest diagnosis to {label} is {ntl[closests[ltn[label]]]} ")
    #print(model.wv.index_to_key)
    #print()
    print(all_concept_map)
    #wv = KeyedVectors.load(model_filename, mmap='r')
    print(model.wv.index_to_key)

    print(all_concept_map_reverse)
    for concept, index in all_concept_map_reverse.items():
        if concept in [443601, 443597,443612, 443611,81902,133810,193782,201826, 40482801,443732,443731,313217,42872402,437827,313217]:
            sims = model.wv.most_similar(str(index), topn=10)
            print(f"The closest diagnoses to {concept} are {all_concept_map[int(sims[0][0])]} and {all_concept_map[int(sims[1][0])]} ")



if __name__ == '__main__':
    pass
    # embedding_filename = "wtv_data_window_90d_20epochs_wv_big100"
    # plot_embedding(f"{embedding_filename}_model")
    # plot_embedding("wtv_data_window_90d_20epochs_model")
