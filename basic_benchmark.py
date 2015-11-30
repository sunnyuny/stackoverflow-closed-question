import competition_utilities as cu
import features
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

train_file = "train-sample.csv"
full_train_file = "train_data.csv"
test_file = "test_data.csv"
submission_file = "basic_benchmark.csv"

feature_names = [ "BodyLength"
                , "NumTags"
                , "OwnerUndeletedAnswerCountAtPostTime"
                , "ReputationAtPostCreation"
                , "TitleLength"
                , "UserAge"
                ]

def main():
    print("Reading the data")
    data = cu.get_dataframe(train_file)

    print cu.get_closed_count('train_data.csv')
    print cu.get_open_count('train_data.csv')

    print("Extracting features")
    fea = features.extract_features(feature_names, data)

    # print("Training random forest")
    # rf = RandomForestClassifier(n_estimators=50, verbose=2, n_jobs=-1)
    # rf.fit(fea, data["OpenStatus"])

    print("Training decision tree")
    dt = DecisionTreeClassifier()
    dt.fit(fea, data["OpenStatus"])

    print("Reading test file and making predictions")
    data = cu.get_dataframe(test_file)
    test_features = features.extract_features(feature_names, data)
    probs = dt.predict_proba(test_features)
    # print cu.get_closed_count('test_data.csv')
    # print cu.get_open_count('test_data.csv')

    print("Calculating priors and updating posteriors")
    new_priors = cu.get_priors(full_train_file)
    old_priors = cu.get_priors(train_file)
    probs = cu.cap_and_update_priors(old_priors, probs, new_priors, 0.001)
    #
    # print("Saving submission to %s" % submission_file)
    # cu.write_submission(submission_file, probs)
    actual = cu.get_actual(data["OpenStatus"])
    print(cu.get_log_loss(actual, probs, 10**(-15)))
if __name__=="__main__":
    main()
