// The path of the read file
#define READ_PATH "C:\\Users\\tavbe\\Desktop\\מחשוב מקבילי ומבוזר\\data2.txt"

// The path of the answer file, where the program write it
#define WRITE_PATH "C:\\Users\\tavbe\\Desktop\\מחשוב מקבילי ומבוזר\\output.txt"

// Constants
#define TERMINATION_TAG 999
#define MAX_POINTS 500000
#define MIN_POINTS 100000
#define MAX_DIM 20
#define MAX_ALPHA_CHECK 100
#define MAX_ITERATION 1000

struct Point {
	float* coords;
	int group;
};

struct Answer {
	float* w;
	float q;
	float a;
	bool foundAnswer;
};



// Methods that the program uses
int sign(int k, float *w, struct Point p);

Point* readFromFile(int *n, int *k, float *a_0, float *a_max, int *limit, float *qc, struct Point *points);

void writeToFile(struct Answer *minAnswer, int k);

void sendPoints(struct Point *p, struct Point *points, int n, int k, int numOfProc, MPI_Datatype *PointMPIType);

void recivePoints(struct Point *p, struct Point *points, int n, int k, int numOfProc, MPI_Datatype *PointMPIType);

void initMinAnswer(struct Answer *answer, struct Answer *minAnswer, int k);