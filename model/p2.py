#import the libraries
from pickle import load

f = open("gpa.pkl","rb")
model = load(f)
f.close()

#prediction
d = [[18,8.191218545250186,0,0,1,1,0,0,0,1.0]]
ans = model.predict(d)
print("GPA : ",ans[0])