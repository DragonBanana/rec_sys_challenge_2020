from AWS.Repository import Repository


class RecommenderRepository:
    recommender_repositories = {}

    dataset_repositories = {}

    def __init__(self):
        recommender_repositories = ['slim-bpr-recommender',
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
                                    'matrix-factorization-bpr-recommender']
        dataset_repositories = ['twitter-2020-dataset',
                                'twitter-2020-cached-dataset']
        for repository in recommender_repositories:
            self.recommender_repositories[repository] = Repository(bucket_name=repository)
        for repository in dataset_repositories:
            self.dataset_repositories[repository] = Repository(bucket_name=repository)

    def sync_all(self):
        self.sync_recommenders()
        self.sync_dataset()

    def sync_recommenders(self):
        for repository in self.recommender_repositories.values():
            repository.sync_all()

    def sync_dataset(self):
        for repository in self.dataset_repositories.values():
            repository.sync_all()

    def sync_repository(self, repository_name):
        if repository_name in self.repositories.keys():
            self.repositories[repository_name].sync_all()
        else:
            raise Exception(f'Repository "{repository_name}"')
