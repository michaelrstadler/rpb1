import unittest
import numpy as np
import tempfile
import cnn_models.siamese_cnn as sm
import cnn_models.evaluate_models as ev
import os
from flymovie.load_save import save_pickle


class TestEvaluateModels(unittest.TestCase):

    #-----------------------------------------------------------------------
    def test_embed_images(self):
        with tempfile.TemporaryDirectory() as tempdir:
            for i in range(4):
                im = np.random.random((20,100,100))
                save_pickle(im, os.path.join(tempdir, 'abc_' + str(i*0.1) 
                        + '_' + str(i*10) + '_' + str(i) + '.pkl'))

            base = sm.make_base_cnn_3d(image_shape=(20,100,100), nlayers=18)
            embedding = sm.make_embedding(base)

            im_embeddings, params = ev.embed_images(tempdir, embedding)

        self.assertTrue(np.array_equal(im_embeddings.shape, (4,256)), 'Wrong size of im_embeddings')
        self.assertTrue(np.array_equal(params.shape, (4,2)), 'Wrong size of params')
        for c in range(params.shape[1]):
            self.assertAlmostEqual(np.mean(params[:,c]), 0, 3, 'Mean should be ~zero')
            self.assertAlmostEqual(np.std(params[:,c]), 1, 3, 'Std should be ~1')

    #-----------------------------------------------------------------------
    def test_rank_embeddingdist_matchedpairs(self):
        e1 = np.random.random((20,256))
         
        ranks = ev.rank_embeddingdist_matchedpairs(e1, e1)
        self.assertTrue(np.array_equal(ranks, np.repeat(0, 20)), 'All ranks should be 0.')
        self.assertEqual(np.mean(ranks), 0, "mean rank should be 0.")
        e_shuff = e1.copy()
        np.random.shuffle(e_shuff)
        ranks = ev.rank_embeddingdist_matchedpairs(e1, e_shuff)
        self.assertGreater(np.mean(ranks), 0, 'Ranks should be >0 after shuffling.')


if __name__ == '__main__':
	unittest.main()