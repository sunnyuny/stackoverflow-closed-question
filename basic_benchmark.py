import competition_utilities as cu
import features
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

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
    train_data = cu.get_dataframe(train_file)

    print("Extracting features")
    train_features = features.extract_features(feature_names, train_data)

    print("Reading test file and making predictions")
    test_data = cu.get_dataframe(test_file)
    test_features = features.extract_features(feature_names, test_data)


    # print("Training random forest")
    # rf = RandomForestClassifier(n_estimators=50, verbose=2, n_jobs=-1)
    # rf.fit(train_features, train_data["OpenStatus"])
    # probs = rf.predict_proba(test_features)

    # print("Training decision tree")
    # dt = DecisionTreeClassifier()
    # dt.fit(train_features, train_data["OpenStatus"])
    # probs = dt.predict_proba(test_features)

    # print("Training adaboost")
    # ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=1.5, algorithm="SAMME").fit(train_features, train_data["OpenStatus"])
    # probs = ada.predict_proba(test_features)

    print("Training nearest neighbors")
    scaler = preprocessing.StandardScaler().fit(train_features)
    train_features_scaled = scaler.transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    nbrs = KNeighborsClassifier(n_neighbors=10).fit(train_features_scaled, train_data["OpenStatus"])
    probs = nbrs.predict_proba(test_features_scaled)

    print("Calculating priors and updating posteriors")
    new_priors = cu.get_priors(full_train_file)
    old_priors = cu.get_priors(train_file)
    probs = cu.cap_and_update_priors(old_priors, probs, new_priors, 0.001)
    actual = cu.get_actual(test_data["OpenStatus"])
    print(cu.get_log_loss(actual, probs, 10**(-15)))
if __name__=="__main__":
    main()
