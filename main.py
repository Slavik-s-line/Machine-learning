import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")

df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df2[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])

df['has_mobile'] = df['has_mobile'].apply(int)
df2['has_mobile'] = df2['has_mobile'].apply(int)

to_del = ['bdate', 'followers_count', 'graduation', 'education_form',
          'relation', 'life_main', 'people_main', 'city', 'last_seen',
          'occupation_name', 'career_start', 'career_end']

df.drop(to_del, axis=1, inplace=True)
df2.drop(to_del, axis=1, inplace=True)

# ---------------------------------------------------------------------


def fill_sex(sex):
    if sex == 2:
        return 1
    return 0


df['sex'] = df['sex'].apply(fill_sex)
df2['sex'] = df2['sex'].apply(fill_sex)

# ---------------------------------------------------------------------


def education(status):
    if status == "Undergraduate applicant":
        return 0
    if status == "Student (Bachelor's)":
        return 1
    if status == "Alumnus (Bachelor's)":
        return 2
    if status == "Student (Master's)":
        return 3
    if status == "Alumnus (Master's)":
        return 4
    if status == "Student (Specialist)":
        return 5
    if status == "Alumnus (Specialist)":
        return 6
    if status == "Candidate of Sciences" or status == "PhD":
        return 7


df['education_status'] = df['education_status'].apply(education)
df2['education_status'] = df2['education_status'].apply(education)

# ---------------------------------------------------------------------


def list_of_langs(langs):
    if len(langs.split(";")) == 1:
        return "1 lang"
    elif len(langs.split(";")) == 2:
        return "2 lang"
    else:
        return "More 3 lang"


df['langs'] = df['langs'].apply(list_of_langs)
df2['langs'] = df2['langs'].apply(list_of_langs)
df[list(pd.get_dummies(df['langs']).columns)] = pd.get_dummies(df['langs'])
df2[list(pd.get_dummies(df['langs']).columns)] = pd.get_dummies(df['langs'])
df.drop('langs', axis=1, inplace=True)
df2.drop('langs', axis=1, inplace=True)
df.info()

# ---------------------------------------------------------------------


def occupation(type):
    if type == "university":
        return 1
    return 0


df['occupation_type'] = df['occupation_type'].apply(occupation)
df2['occupation_type'] = df2['occupation_type'].apply(occupation)
df.info()

# ---------------------------------------------------------------------

X_train = df.drop('result', axis=1)
y_train = df['result']
X_test = df2

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
ID = df2['id']
result = pd.DataFrame({'id': ID, 'result': y_pred})
result.to_csv('answer.csv', index=False)
