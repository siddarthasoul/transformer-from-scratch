student_grade = {}

# name of the student 
def add_student(name,garde):
    student_grade[name] = garde
    print(f"enter {name} of the student {garde}")

#update the student
def update_student(name,garde):
    if name in student_grade:
        student_grade[name]= garde
        print(f" {name} of the student is updated with marks{garde}")

    else:
        print("not data found!!!!!!")

#delete the student

def del_student(name):
    if name in student_grade:
        del student_grade[name]
        print(f"{name} of the student is successfully deleted")
    else:
        print("no data found!!!!!!!")

#view all students
def display_all_student():
       if student_grade:
           for name,grade in student_grade.items():
               print(f"{name} : {grade}")
       else:
               print("no student found")

def main():
    print("start")
    while True:
        print("\n student grade management system")
        print("1. add student")
        print("2. update student")
        print("3. delete student")
        print("4. view student")
        print("5. Exit")
      
        choice = int(input("enter your choice = "))
        if choice == 1:
          name = input("Enter student name = ")
          grade = int(input("Enter student grade ="))
          add_student(name,grade)
    
        elif choice == 2:
          name = input("Enter student name = ")
          grade = int(input("Enter student grade ="))
          update_student(name,grade)

        elif choice == 3:
          name = input("Enter student name = ")
          del_student(name)

        elif choice == 4:
           display_all_student()

        elif choice == 5:
            print("closing the program....")

        else:
            print("invalid choice")



main()