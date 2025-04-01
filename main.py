from Bayesian import Bayesian


bayes = Bayesian("dataset.json", "dataset.json")

bayes.train()
bayes.save_model("bayes.json")
bayes2 = Bayesian("dataset.json", "dataset.json")
bayes2.load_model("bayes.json")
bayes2.test()