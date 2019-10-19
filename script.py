import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz
fig, ax = plt.subplots()
david_ortiz['type']=david_ortiz['type'].map({'S':1,'B':0})
david_ortiz=david_ortiz.dropna(subset=  ['type','plate_x','plate_z'])
plt.scatter(x=david_ortiz.plate_x,y=david_ortiz.plate_z,     c=david_ortiz.type, cmap=plt.cm.coolwarm, alpha=0.5)
training_set, validation_set=         train_test_split(david_ortiz, random_state=1)
classifier=SVC(kernel='rbf', gamma=100, C=100)
classifier.fit(training_set[['plate_x','plate_z']],   training_set['type'])
draw_boundary(ax, classifier)
accuracy= classifier.score(training_set[['plate_x','plate_z']], training_set['type'])
print(accuracy)
ax.set_ylim(-2,6)
ax.set_xlim(-3,3)
plt.show()


