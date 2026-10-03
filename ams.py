import csv
import os
from datetime import datetime
from pathlib import Path

FILE_NAME = Path(__file__).resolve().parent / "2018-2019_Daily_Attendance_20240429.csv"


# -----------------------------------------
# CHECK FILE
# -----------------------------------------
def check_file():

    if not os.path.exists(FILE_NAME):
        print("CSV file not found!")
        print("Please put the CSV file in the same folder.")
        return False

    return True


# -----------------------------------------
# VIEW ALL RECORDS
# -----------------------------------------
def view_all_records():

    if not check_file():
        input("\nPress Enter to continue...")
        return

    with open(FILE_NAME, "r", newline="", encoding="utf-8-sig") as f:

        reader = csv.DictReader(f)

        print("\n----- ALL ATTENDANCE RECORDS -----")
        print("School DBN | Date | Enrolled | Absent | Present | Released")
        print("-" * 75)

        for row in reader:

            print(
                row["School DBN"], "|",
                row["Date"], "|",
                row["Enrolled"], "|",
                row["Absent"], "|",
                row["Present"], "|",
                row["Released"]
            )

    input("\nPress Enter to go back to the main menu...")


# -----------------------------------------
# SEARCH BY SCHOOL DBN
# -----------------------------------------
def search_by_school():

    school = input("Enter School DBN: ").upper()

    f = open(FILE_NAME, "r")
    reader = csv.DictReader(f)

    found = False

    print("\n----- SCHOOL ATTENDANCE -----")
    print("School DBN | Date | Enrolled | Absent | Present")
    print("-" * 65)

    for row in reader:

        if row["School DBN"] == school:

            print(
                row["School DBN"], "|",
                row["Date"], "|",
                row["Enrolled"], "|",
                row["Absent"], "|",
                row["Present"]
            )

            found = True

    f.close()

    if found == False:
        print("School DBN not found!")


# -----------------------------------------
# SEARCH BY DATE
# -----------------------------------------
def search_by_date():

    print("Example date: 20180905 or 2018-09-05")
    search_date = input("Enter Date: ").strip().replace("-", "")

    f = open(FILE_NAME, "r")
    reader = csv.DictReader(f)

    found = False

    print("\n----- ATTENDANCE BY DATE -----")
    print("School DBN | Enrolled | Absent | Present")
    print("-" * 60)

    for row in reader:

        if row["Date"].strip().replace("-", "") == search_date:

            print(
                row["School DBN"], "|",
                row["Enrolled"], "|",
                row["Absent"], "|",
                row["Present"]
            )

            found = True

    f.close()

    if found == False:
        print("No attendance records found for this date!")


# -----------------------------------------
# ATTENDANCE PERCENTAGE
# -----------------------------------------
def attendance_percentage():

    school = input("Enter School DBN: ").upper()

    total_enrolled = 0
    total_present = 0

    f = open(FILE_NAME, "r")
    reader = csv.DictReader(f)

    found = False

    for row in reader:

        if row["School DBN"] == school:

            total_enrolled += int(row["Enrolled"])
            total_present += int(row["Present"])

            found = True

    f.close()

    if found == False:

        print("School DBN not found!")

    else:

        percentage = (total_present / total_enrolled) * 100

        print("\n----- ATTENDANCE REPORT -----")
        print("School DBN:", school)
        print("Total Enrolled:", total_enrolled)
        print("Total Present:", total_present)
        print("Attendance Percentage:", round(percentage, 2), "%")


# -----------------------------------------
# TOTAL PRESENT STUDENTS
# -----------------------------------------
def total_present_students():

    total = 0

    f = open(FILE_NAME, "r")
    reader = csv.DictReader(f)

    for row in reader:

        total += int(row["Present"])

    f.close()

    print("\nTotal Present Students:", total)


# -----------------------------------------
# TOTAL ABSENT STUDENTS
# -----------------------------------------
def total_absent_students():

    total = 0

    f = open(FILE_NAME, "r")
    reader = csv.DictReader(f)

    for row in reader:

        total += int(row["Absent"])

    f.close()

    print("\nTotal Absent Students:", total)


# -----------------------------------------
# HIGHEST ATTENDANCE
# -----------------------------------------
def highest_attendance():

    highest = -1
    best_record = None

    f = open(FILE_NAME, "r")
    reader = csv.DictReader(f)

    for row in reader:

        present = int(row["Present"])

        if present > highest:

            highest = present
            best_record = row

    f.close()

    print("\n----- HIGHEST ATTENDANCE RECORD -----")

    print("School DBN:", best_record["School DBN"])
    print("Date:", best_record["Date"])
    print("Enrolled:", best_record["Enrolled"])
    print("Present:", best_record["Present"])
    print("Absent:", best_record["Absent"])


# -----------------------------------------
# LOWEST ATTENDANCE
# -----------------------------------------
def lowest_attendance():

    lowest = None
    low_record = None

    f = open(FILE_NAME, "r")
    reader = csv.DictReader(f)

    for row in reader:

        present = int(row["Present"])

        if lowest is None or present < lowest:

            lowest = present
            low_record = row

    f.close()

    print("\n----- LOWEST ATTENDANCE RECORD -----")

    print("School DBN:", low_record["School DBN"])
    print("Date:", low_record["Date"])
    print("Enrolled:", low_record["Enrolled"])
    print("Present:", low_record["Present"])
    print("Absent:", low_record["Absent"])


# -----------------------------------------
# ADD NEW ATTENDANCE RECORD
# -----------------------------------------
def add_record():

    school = input("Enter School DBN: ").upper()
    raw_date = input("Enter Date (YYYYMMDD or YYYY-MM-DD): ").strip()
    try:
        record_date = datetime.strptime(raw_date.replace("-", ""), "%Y%m%d").strftime("%Y%m%d")
        enrolled, absent, present, released = [
            int(input(prompt))
            for prompt in (
                "Enter Total Enrolled Students: ",
                "Enter Absent Students: ",
                "Enter Present Students: ",
                "Enter Released Students: ",
            )
        ]
        if min(enrolled, absent, present, released) < 0:
            raise ValueError("Attendance counts must be non-negative.")
        if present + absent + released != enrolled:
            raise ValueError("Present + absent + released must equal enrolled.")
    except ValueError as error:
        print("Invalid record:", error)
        return

    with open(FILE_NAME, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([school, record_date, enrolled, absent, present, released])

    print("New attendance record added successfully!")


# -----------------------------------------
# MAIN MENU
# -----------------------------------------
def main():
    while True:
        print("\n============================================")
        print(" SCHOOL ATTENDANCE MANAGEMENT SYSTEM")
        print("============================================")
        print("1. View All Attendance Records")
        print("2. Search Attendance by School DBN")
        print("3. Search Attendance by Date")
        print("4. Calculate Attendance Percentage")
        print("5. View Total Present Students")
        print("6. View Total Absent Students")
        print("7. Find Highest Attendance Record")
        print("8. Find Lowest Attendance Record")
        print("9. Add New Attendance Record")
        print("10. Exit")

        choice = input("\nEnter your choice: ")
        if choice == "1":
            view_all_records()
        elif choice == "2":
            search_by_school()
        elif choice == "3":
            search_by_date()
        elif choice == "4":
            attendance_percentage()
        elif choice == "5":
            total_present_students()
        elif choice == "6":
            total_absent_students()
        elif choice == "7":
            highest_attendance()
        elif choice == "8":
            lowest_attendance()
        elif choice == "9":
            add_record()
        elif choice == "10":
            print("\nThank you for using the system!")
            break
        else:
            print("Invalid choice! Please enter 1 to 10.")


if __name__ == "__main__":
    main()