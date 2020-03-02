from AWS.Repository import Repository


class RecommenderRepository:

    repositories = {}

    def __init__(self):
        repositories_list = ['slim_bpr_recommender',
                             'slim_elastic_net_recommender',
                             'item_cf_recommender',
                             'item_cbf_recommender',
                             'user_cf_recommender',
                             'user_cbf_recommender',
                             'p3_alpha_recommender',
                             'rp3_beta_recommender',
                             'ials_recommender',
                             'pure_svd_recommender',
                             'funk_svd_recommender',
                             'asy_svd_recommender',
                             'matrix_factorization_bpr_recommender',
                             'dataset']
        for repository in repositories_list:
            self.repositories[repository] = Repository(bucket_name=repository)

    def sync_all(self):
        for repository in self.repositories:
            repository.sync_all()

    def sync_repository(self, repository_name):
        if repository_name in self.repositories.keys():
            self.repositories[repository_name].sync_all()
        else:
            raise Exception(f'Repository "{repository_name}"')
