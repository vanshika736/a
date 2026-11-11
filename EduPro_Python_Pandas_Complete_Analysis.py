# ================================================================
# EDUPRO: INSTRUCTOR PERFORMANCE & COURSE QUALITY EVALUATION
# Python + Pandas + NumPy + Matplotlib + Seaborn
# Dataset: EduPro Online Platform.xlsx
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------
# 1. CONFIGURATION
# ---------------------------
FILE_PATH = "EduPro Online Platform.xlsx"
OUTPUT_DIR = Path("EduPro_Analysis_Output")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="Greens")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

# ---------------------------
# 2. LOAD DATA
# ---------------------------
users = pd.read_excel(FILE_PATH, sheet_name="Users")
teachers = pd.read_excel(FILE_PATH, sheet_name="Teachers")
courses = pd.read_excel(FILE_PATH, sheet_name="Courses")
transactions = pd.read_excel(FILE_PATH, sheet_name="Transactions")

print("\n========== DATASET OVERVIEW ==========")
for name, df in {
    "Users": users,
    "Teachers": teachers,
    "Courses": courses,
    "Transactions": transactions
}.items():
    print(f"{name}: {df.shape[0]} rows x {df.shape[1]} columns")

# ---------------------------
# 3. DATA QUALITY CHECKS
# ---------------------------
print("\n========== DATA QUALITY ==========")

for name, df in {
    "Users": users,
    "Teachers": teachers,
    "Courses": courses,
    "Transactions": transactions
}.items():
    print(f"\n{name} - missing values:")
    print(df.isnull().sum())
    print(f"{name} - duplicate rows: {df.duplicated().sum()}")

# Primary-key duplicate checks
primary_keys = {
    "Users": (users, "UserID"),
    "Teachers": (teachers, "TeacherID"),
    "Courses": (courses, "CourseID"),
    "Transactions": (transactions, "TransactionID")
}

for name, (df, key) in primary_keys.items():
    print(f"{name} duplicate {key}: {df[key].duplicated().sum()}")

# Foreign-key validation
print("\nForeign-key validation:")
print("Transactions with unknown UserID:",
      (~transactions["UserID"].isin(users["UserID"])).sum())
print("Transactions with unknown TeacherID:",
      (~transactions["TeacherID"].isin(teachers["TeacherID"])).sum())
print("Transactions with unknown CourseID:",
      (~transactions["CourseID"].isin(courses["CourseID"])).sum())

# ---------------------------
# 4. DATA CLEANING
# ---------------------------
# Remove exact duplicate rows only. Do not silently remove duplicate IDs.
users = users.drop_duplicates().copy()
teachers = teachers.drop_duplicates().copy()
courses = courses.drop_duplicates().copy()
transactions = transactions.drop_duplicates().copy()

# Convert numeric fields safely
for col in ["Age", "YearsOfExperience", "TeacherRating"]:
    teachers[col] = pd.to_numeric(teachers[col], errors="coerce")

courses["CourseRating"] = pd.to_numeric(
    courses["CourseRating"], errors="coerce"
)
transactions["Amount"] = pd.to_numeric(
    transactions["Amount"], errors="coerce"
)

# ---------------------------
# 5. DATA INTEGRATION
# ---------------------------
# Aggregate transaction activity by instructor
teacher_activity = (
    transactions.groupby("TeacherID")
    .agg(
        TotalTransactions=("TransactionID", "count"),
        TotalRevenue=("Amount", "sum"),
        UniqueCourses=("CourseID", "nunique"),
        UniqueLearners=("UserID", "nunique")
    )
    .reset_index()
)

# Aggregate transaction activity by course
course_activity = (
    transactions.groupby("CourseID")
    .agg(
        TotalTransactions=("TransactionID", "count"),
        TotalRevenue=("Amount", "sum"),
        NumberOfInstructors=("TeacherID", "nunique"),
        UniqueLearners=("UserID", "nunique")
    )
    .reset_index()
)

# Merge instructor profile and activity
teacher_analysis = teachers.merge(
    teacher_activity, on="TeacherID", how="left"
)

# Merge course information and activity
course_analysis = courses.merge(
    course_activity, on="CourseID", how="left"
)

# Fill missing activity counts with zero
activity_columns_teacher = [
    "TotalTransactions", "TotalRevenue", "UniqueCourses", "UniqueLearners"
]
activity_columns_course = [
    "TotalTransactions", "TotalRevenue",
    "NumberOfInstructors", "UniqueLearners"
]

teacher_analysis[activity_columns_teacher] = (
    teacher_analysis[activity_columns_teacher].fillna(0)
)
course_analysis[activity_columns_course] = (
    course_analysis[activity_columns_course].fillna(0)
)

# Combined transaction-level table for exploration
combined = (
    transactions
    .merge(
        teachers[["TeacherID", "TeacherName", "Age", "Gender",
                  "Expertise", "YearsOfExperience", "TeacherRating"]],
        on="TeacherID", how="left", validate="many_to_one"
    )
    .merge(
        courses[["CourseID", "CourseName", "CourseCategory",
                 "CourseLevel", "CourseRating"]],
        on="CourseID", how="left", validate="many_to_one"
    )
)

print("\n========== INTEGRATED DATA ==========")
print(combined.head())

# ---------------------------
# 6. KEY PERFORMANCE INDICATORS
# ---------------------------
total_users = users["UserID"].nunique()
total_teachers = teachers["TeacherID"].nunique()
total_courses = courses["CourseID"].nunique()
total_transactions = transactions["TransactionID"].nunique()

average_teacher_rating = teachers["TeacherRating"].mean()
average_course_rating = courses["CourseRating"].mean()
average_experience = teachers["YearsOfExperience"].mean()
total_revenue = transactions["Amount"].sum()

kpi = pd.DataFrame({
    "KPI": [
        "Total Users",
        "Total Instructors",
        "Total Courses",
        "Total Transactions",
        "Average Teacher Rating",
        "Average Course Rating",
        "Average Instructor Experience (Years)",
        "Total Recorded Transaction Amount"
    ],
    "Value": [
        total_users,
        total_teachers,
        total_courses,
        total_transactions,
        average_teacher_rating,
        average_course_rating,
        average_experience,
        total_revenue
    ]
})

print("\n========== KPI SUMMARY ==========")
print(kpi)

# ---------------------------
# 7. INSTRUCTOR PERFORMANCE
# ---------------------------
instructor_leaderboard = teacher_analysis.sort_values(
    by=["TeacherRating", "TotalTransactions"],
    ascending=[False, False]
)

print("\n========== TOP 10 INSTRUCTORS ==========")
print(
    instructor_leaderboard[
        ["TeacherID", "TeacherName", "Expertise",
         "YearsOfExperience", "TeacherRating", "TotalTransactions"]
    ].head(10)
)

print("\n========== LOWEST 10 INSTRUCTOR RATINGS ==========")
print(
    teacher_analysis.sort_values("TeacherRating")[
        ["TeacherID", "TeacherName", "Expertise",
         "YearsOfExperience", "TeacherRating", "TotalTransactions"]
    ].head(10)
)

# Rating tiers: descriptive groups; adjust thresholds if required
teacher_analysis["RatingTier"] = pd.cut(
    teacher_analysis["TeacherRating"],
    bins=[-np.inf, 2.5, 3.75, np.inf],
    labels=["Low", "Medium", "High"],
    right=True
)

rating_tier_summary = (
    teacher_analysis.groupby("RatingTier", observed=False)
    .agg(
        InstructorCount=("TeacherID", "nunique"),
        AverageRating=("TeacherRating", "mean"),
        AverageTransactions=("TotalTransactions", "mean")
    )
    .reset_index()
)

# ---------------------------
# 8. EXPERIENCE VS PERFORMANCE
# ---------------------------
experience_teacher_corr = teachers[
    ["YearsOfExperience", "TeacherRating"]
].corr(method="pearson").iloc[0, 1]

experience_bins = [-np.inf, 2, 5, 10, 20, np.inf]
experience_labels = [
    "0-2 years", "3-5 years", "6-10 years",
    "11-20 years", "20+ years"
]

teacher_analysis["ExperienceBand"] = pd.cut(
    teacher_analysis["YearsOfExperience"],
    bins=experience_bins,
    labels=experience_labels
)

experience_summary = (
    teacher_analysis.groupby("ExperienceBand", observed=False)
    .agg(
        InstructorCount=("TeacherID", "count"),
        AverageTeacherRating=("TeacherRating", "mean"),
        AverageTransactions=("TotalTransactions", "mean")
    )
    .reset_index()
)

print("\nExperience vs Teacher Rating correlation:",
      round(experience_teacher_corr, 3))
print(experience_summary)

# ---------------------------
# 9. COURSE QUALITY ANALYSIS
# ---------------------------
category_summary = (
    course_analysis.groupby("CourseCategory")
    .agg(
        NumberOfCourses=("CourseID", "nunique"),
        AverageCourseRating=("CourseRating", "mean"),
        TotalTransactions=("TotalTransactions", "sum"),
        TotalRevenue=("TotalRevenue", "sum")
    )
    .reset_index()
    .sort_values("AverageCourseRating", ascending=False)
)

level_summary = (
    course_analysis.groupby("CourseLevel")
    .agg(
        NumberOfCourses=("CourseID", "nunique"),
        AverageCourseRating=("CourseRating", "mean"),
        TotalTransactions=("TotalTransactions", "sum")
    )
    .reset_index()
)

print("\n========== COURSE CATEGORY SUMMARY ==========")
print(category_summary)

print("\n========== COURSE LEVEL SUMMARY ==========")
print(level_summary)

# ---------------------------
# 10. EXPERTISE-WISE ANALYSIS
# ---------------------------
expertise_summary = (
    teacher_analysis.groupby("Expertise")
    .agg(
        InstructorCount=("TeacherID", "nunique"),
        AverageTeacherRating=("TeacherRating", "mean"),
        AverageExperience=("YearsOfExperience", "mean"),
        TotalTransactions=("TotalTransactions", "sum"),
        AverageTransactions=("TotalTransactions", "mean")
    )
    .reset_index()
    .sort_values("AverageTeacherRating", ascending=False)
)

print("\n========== EXPERTISE PERFORMANCE ==========")
print(expertise_summary)

# ---------------------------
# 11. GENDER AND AGE ANALYSIS
# ---------------------------
gender_summary = (
    teacher_analysis.groupby("Gender")
    .agg(
        InstructorCount=("TeacherID", "nunique"),
        AverageTeacherRating=("TeacherRating", "mean"),
        AverageExperience=("YearsOfExperience", "mean")
    )
    .reset_index()
)

teacher_analysis["AgeGroup"] = pd.cut(
    teacher_analysis["Age"],
    bins=[0, 25, 35, 45, 55, np.inf],
    labels=["18-25", "26-35", "36-45", "46-55", "56+"]
)

age_summary = (
    teacher_analysis.groupby("AgeGroup", observed=False)
    .agg(
        InstructorCount=("TeacherID", "count"),
        AverageTeacherRating=("TeacherRating", "mean")
    )
    .reset_index()
)

# ---------------------------
# 12. ENROLLMENT / TRANSACTION ANALYSIS
# ---------------------------
teacher_rating_enrollment_corr = teacher_analysis[
    ["TeacherRating", "TotalTransactions"]
].corr(method="pearson").iloc[0, 1]

top_enrollment_instructors = teacher_analysis.sort_values(
    "TotalTransactions", ascending=False
).head(10)

# Important: transaction volume is not necessarily unique enrollments.
# Each transaction row is treated as an enrollment proxy in this project.

# ---------------------------
# 13. TEACHER RATING VS COURSE RATING
# ---------------------------
# Use distinct teacher-course pairs to avoid counting every transaction.
teacher_course_pairs = (
    transactions[["TeacherID", "CourseID"]]
    .drop_duplicates()
    .merge(
        teachers[["TeacherID", "TeacherRating"]],
        on="TeacherID", how="left"
    )
    .merge(
        courses[["CourseID", "CourseRating"]],
        on="CourseID", how="left"
    )
)

teacher_course_corr = teacher_course_pairs[
    ["TeacherRating", "CourseRating"]
].corr(method="pearson").iloc[0, 1]

print("\nTeacher Rating vs Course Rating correlation across linked pairs:",
      round(teacher_course_corr, 3))
print("Interpret cautiously: CourseRating is stored at course level, not "
      "for each instructor-course learner evaluation.")

# ---------------------------
# 14. VISUALIZATIONS
# ---------------------------
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# 14.1 Teacher rating distribution
plt.figure(figsize=(8, 5))
sns.histplot(teachers["TeacherRating"].dropna(), bins=10, kde=True)
plt.title("Distribution of Instructor Ratings")
plt.xlabel("Teacher Rating")
plt.ylabel("Number of Instructors")
save_plot("01_teacher_rating_distribution.png")

# 14.2 Experience vs teacher rating
plt.figure(figsize=(8, 5))
sns.regplot(
    data=teachers, x="YearsOfExperience", y="TeacherRating",
    scatter_kws={"alpha": 0.8}
)
plt.title("Teaching Experience vs Instructor Rating")
plt.xlabel("Years of Experience")
plt.ylabel("Teacher Rating")
save_plot("02_experience_vs_teacher_rating.png")

# 14.3 Expertise-wise ratings
plt.figure(figsize=(10, 5))
sns.barplot(
    data=expertise_summary,
    x="Expertise", y="AverageTeacherRating"
)
plt.title("Average Instructor Rating by Expertise")
plt.xticks(rotation=35, ha="right")
plt.xlabel("Expertise")
plt.ylabel("Average Teacher Rating")
save_plot("03_expertise_performance.png")

# 14.4 Course rating by category
plt.figure(figsize=(10, 5))
sns.barplot(
    data=category_summary,
    x="CourseCategory", y="AverageCourseRating"
)
plt.title("Average Course Rating by Category")
plt.xticks(rotation=35, ha="right")
plt.xlabel("Course Category")
plt.ylabel("Average Course Rating")
save_plot("04_course_rating_by_category.png")

# 14.5 Course rating by level
plt.figure(figsize=(8, 5))
sns.barplot(
    data=level_summary,
    x="CourseLevel", y="AverageCourseRating"
)
plt.title("Average Course Rating by Course Level")
plt.xlabel("Course Level")
plt.ylabel("Average Course Rating")
save_plot("05_course_rating_by_level.png")

# 14.6 Top instructors by rating
top10_rating = instructor_leaderboard.head(10)
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top10_rating,
    x="TeacherRating", y="TeacherName"
)
plt.title("Top 10 Instructors by Rating")
plt.xlabel("Teacher Rating")
plt.ylabel("Instructor")
save_plot("06_top_instructors_by_rating.png")

# 14.7 Top instructors by transaction volume
top10_tx = top_enrollment_instructors
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top10_tx,
    x="TotalTransactions", y="TeacherName"
)
plt.title("Top 10 Instructors by Transaction Count")
plt.xlabel("Transaction Count")
plt.ylabel("Instructor")
save_plot("07_top_instructors_by_transactions.png")

# 14.8 Course category x course level heatmap
heatmap_data = course_analysis.pivot_table(
    index="CourseCategory",
    columns="CourseLevel",
    values="CourseRating",
    aggfunc="mean"
)
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap="Greens", fmt=".2f")
plt.title("Course Quality Heatmap: Category vs Level")
plt.xlabel("Course Level")
plt.ylabel("Course Category")
save_plot("08_course_quality_heatmap.png")

# 14.9 Rating correlation heatmap (teacher-level variables)
plt.figure(figsize=(7, 5))
corr_matrix = teacher_analysis[
    ["Age", "YearsOfExperience", "TeacherRating", "TotalTransactions"]
].corr()
sns.heatmap(corr_matrix, annot=True, cmap="Greens", fmt=".2f")
plt.title("Instructor Performance Correlation Matrix")
save_plot("09_instructor_correlation_heatmap.png")

# ---------------------------
# 15. EXPORT RESULTS TO EXCEL
# ---------------------------
output_excel = OUTPUT_DIR / "EduPro_Analysis_Results.xlsx"

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    kpi.to_excel(writer, sheet_name="KPI Summary", index=False)
    teacher_analysis.to_excel(writer, sheet_name="Instructor Analysis", index=False)
    course_analysis.to_excel(writer, sheet_name="Course Analysis", index=False)
    instructor_leaderboard.to_excel(writer, sheet_name="Instructor Leaderboard", index=False)
    experience_summary.to_excel(writer, sheet_name="Experience Analysis", index=False)
    expertise_summary.to_excel(writer, sheet_name="Expertise Analysis", index=False)
    category_summary.to_excel(writer, sheet_name="Category Analysis", index=False)
    level_summary.to_excel(writer, sheet_name="Course Level Analysis", index=False)
    gender_summary.to_excel(writer, sheet_name="Gender Analysis", index=False)
    age_summary.to_excel(writer, sheet_name="Age Analysis", index=False)
    rating_tier_summary.to_excel(writer, sheet_name="Rating Tier Analysis", index=False)
    teacher_course_pairs.to_excel(writer, sheet_name="Teacher Course Pairs", index=False)
    combined.to_excel(writer, sheet_name="Integrated Data", index=False)

print("\n==========================================")
print("EDUPRO ANALYSIS COMPLETED SUCCESSFULLY")
print("Output folder:", OUTPUT_DIR.resolve())
print("Excel results:", output_excel.resolve())
print("==========================================")
