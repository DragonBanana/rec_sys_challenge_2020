from AWS.Repository import Repository


class RecommenderRepository:

    repositories = {}

    def __init__(self):
        repositories_list = ['slim-bpr-recommender',
                             'slim-elastic-net-recommender',
                             'item-cf-recommender',
                             'item-cbf-recommender',
                             'user-cf-recommender',
                             'user-cbf-recommender',
                             'p3-alpha-recommender',
                             'rp3-beta-recommender',
                             'ials-recommender',
                             'pure-svd-recommender',
                             'funk-svd-recommender',
                             'asy-svd-recommender',
                             'matrix-factorization-bpr-recommender',
                             'twitter-2014-dataset',
                             'twitter-2014-cached-dataset']
        for repository in repositories_list:
            self.repositories[repository] = Repository(bucket_name=repository)

    def sync_all(self):
        for repository in self.repositories.values():
            repository.sync_all()

    def sync_repository(self, repository_name):
        if repository_name in self.repositories.keys():
            self.repositories[repository_name].sync_all()
        else:
            raise Exception(f'Repository "{repository_name}"')
