import codecs
import numpy as np
import copy
import time
import random

import pandas as pd

def norm_l1(h, r, t):
    return np.sum(np.fabs(h + r - t))

def norm_l2(h, r, t):
    return np.sum(np.square(h + r - t))

class TransE:
    def __init__(self, entity, relation, triple_list, embedding_dim=50, lr=0.01, margin=1.0, norm=1):
        self.entities = entity
        self.relations = relation
        self.triples = triple_list
        self.dimension = embedding_dim
        self.learning_rate = lr
        self.margin = margin
        self.norm = norm
        self.loss = 0.0

    def data_initialise(self):
        entityVectorList = {}
        relationVectorList = {}
        for entity in self.entities:
            entity_vector = np.random.uniform(-6.0 / np.sqrt(self.dimension), 6.0 / np.sqrt(self.dimension), self.dimension)
            entityVectorList[entity] = entity_vector
        for relation in self.relations:
            relation_vector = np.random.uniform(-6.0 / np.sqrt(self.dimension), 6.0 / np.sqrt(self.dimension), self.dimension)
            relation_vector = self.normalization(relation_vector)
            relationVectorList[relation] = relation_vector
        self.entities = entityVectorList
        self.relations = relationVectorList

    def normalization(self, vector):
        return vector / np.linalg.norm(vector)

    def training_run(self, epochs=1, nbatches=5):
        print('constructing transE matrix ')
        batch_size = int(len(self.triples) / nbatches)
        print("batch size: ", batch_size)
        for epoch in range(epochs): 
            start = time.time()
            self.loss = 0.0
            # Normalise the embedding of the entities to 1
            for entity in self.entities.keys():
                self.entities[entity] = self.normalization(self.entities[entity])
            for batch in range(nbatches):
                batch_samples = random.sample(self.triples, batch_size)
                Tbatch = []
                for sample in batch_samples:
                    corrupted_sample = copy.deepcopy(sample)
                    pr = np.random.random(1)[0]
                    if pr > 0.5:
                        # change the head entity
                        corrupted_sample[0] = random.sample(self.entities.keys(), 1)[0]
                        while corrupted_sample[0] == sample[0]:
                            corrupted_sample[0] = random.sample(self.entities.keys(), 1)[0]
                    else:
                        # change the tail entity
                        corrupted_sample[2] = random.sample(self.entities.keys(), 1)[0]
                        while corrupted_sample[2] == sample[2]:
                            corrupted_sample[2] = random.sample(self.entities.keys(), 1)[0]
                    if (sample, corrupted_sample) not in Tbatch:
                        Tbatch.append((sample, corrupted_sample))
                self.update_triple_embedding(Tbatch)
            end = time.time()
        return self.entities, self.relations

    def update_triple_embedding(self, Tbatch):
        # deepcopy 可以保证，即使list嵌套list也能让各层的地址不同， 即这里copy_entity 和
        # entitles中所有的elements都不同
        copy_entity = copy.deepcopy(self.entities)
        copy_relation = copy.deepcopy(self.relations)
        for correct_sample, corrupted_sample in Tbatch:
            correct_copy_head = copy_entity[correct_sample[0]]
            correct_copy_tail = copy_entity[correct_sample[2]]
            relation_copy = copy_relation[correct_sample[1]]
            corrupted_copy_head = copy_entity[corrupted_sample[0]]
            corrupted_copy_tail = copy_entity[corrupted_sample[2]]
            correct_head = self.entities[correct_sample[0]]
            correct_tail = self.entities[correct_sample[2]]
            relation = self.relations[correct_sample[1]]
            corrupted_head = self.entities[corrupted_sample[0]]
            corrupted_tail = self.entities[corrupted_sample[2]]
            # calculate the distance of the triples
            if self.norm == 1:
                correct_distance = norm_l1(correct_head, relation, correct_tail)
                corrupted_distance = norm_l1(corrupted_head, relation, corrupted_tail)
            else:
                correct_distance = norm_l2(correct_head, relation, correct_tail)
                corrupted_distance = norm_l2(corrupted_head, relation, corrupted_tail)

            loss = self.margin + correct_distance - corrupted_distance
            if loss > 0:
                self.loss += loss
                correct_gradient = 2 * (correct_head + relation - correct_tail)
                corrupted_gradient = 2 * (corrupted_head + relation - corrupted_tail)
                if self.norm == 1:
                    for i in range(len(correct_gradient)):
                        if correct_gradient[i] > 0:
                            correct_gradient[i] = 1
                        else:
                            correct_gradient[i] = -1

                        if corrupted_gradient[i] > 0:
                            corrupted_gradient[i] = 1
                        else:
                            corrupted_gradient[i] = -1

                correct_copy_head -= self.learning_rate * correct_gradient
                relation_copy -= self.learning_rate * correct_gradient
                correct_copy_tail -= -1 * self.learning_rate * correct_gradient

                relation_copy -= -1 * self.learning_rate * corrupted_gradient
                if correct_sample[0] == corrupted_sample[0]:
                    # if corrupted_triples replaces the tail entity, the head entity's embedding need to be updated twice
                    correct_copy_head -= -1 * self.learning_rate * corrupted_gradient
                    corrupted_copy_tail -= self.learning_rate * corrupted_gradient
                elif correct_sample[2] == corrupted_sample[2]:
                    # if corrupted_triples replaces the head entity, the tail entity's embedding need to be updated twice
                    corrupted_copy_head -= -1 * self.learning_rate * corrupted_gradient
                    correct_copy_tail -= self.learning_rate * corrupted_gradient
                # print(correct_copy_head)
                # normalising these new embedding vector, instead of normalising all the embedding together
                copy_entity[correct_sample[0]] = self.normalization(correct_copy_head)
                copy_entity[correct_sample[2]] = self.normalization(correct_copy_tail)
                if correct_sample[0] == corrupted_sample[0]:
                    # if corrupted_triples replace the tail entity, update the tail entity's embedding
                    copy_entity[corrupted_sample[2]] = self.normalization(corrupted_copy_tail)
                elif correct_sample[2] == corrupted_sample[2]:
                    # if corrupted_triples replace the head entity, update the head entity's embedding
                    copy_entity[corrupted_sample[0]] = self.normalization(corrupted_copy_head)
                # the paper mention that the relation's embedding don't need to be normalised
                copy_relation[correct_sample[1]] = relation_copy
                # copy_relation[correct_sample[1]] = self.normalization(relation_copy)
        self.entities = copy_entity
        self.relations = copy_relation

