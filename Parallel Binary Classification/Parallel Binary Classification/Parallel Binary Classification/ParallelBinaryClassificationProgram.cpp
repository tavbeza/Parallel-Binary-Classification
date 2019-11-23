#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include "mpi.h"
#include "ParallelBinaryClassification.h"

using namespace std;

int main(int argc, char **argv)
{
	int root = 0, myrank, numOfProc, i, errorCode = MPI_ERR_COMM;
	int n, k, limit, iteration = 0, numOfWrong = 0, result, numOfAlpha, alphaSendLeft;
	bool correct = true;
	float a, a_0, a_max, qc, *w, q = 0, *arrayOfAlpha = 0;
	struct Point *points = NULL, p;
	struct Answer answer, minAnswer;
	MPI_Status status;
	MPI_Datatype AnswerMPIType, PointMPIType;
	MPI_Datatype answerType[4] = { MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_C_BOOL }, pointType[2] = { MPI_FLOAT, MPI_INT };
	MPI_Aint answerDisp[4], pointDisp[2];
	
	// Variables to check the runtime
	//double t1 = 0, t2 = 0;

	// Init MPI program
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProc);

	// Exit if launced only one process
	if (numOfProc == 1)
		MPI_Abort(MPI_COMM_WORLD, errorCode);

	std::cout << std::fixed;

	// read all points from the file
	if (myrank == root)
	{
		// t1 = start time
		//t1 = MPI_Wtime();

		// start read from file
		points = readFromFile(&n, &k, &a_0, &a_max, &limit, &qc, points);
	}
	
	// Root send the first variables to childs
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&limit, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&qc, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&a_max, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&a_0, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// Childs allocation for points array
	if (myrank != root)
	{
		points = (struct Point*)malloc(n * sizeof(struct Point));
		#pragma omp parallel for
		for (i = 0; i < n; i++)
		{
			points[i].coords = (float*)malloc((k + 1) *  sizeof(float));
		}
	}

	// Init Vector W
	w = (float*)calloc(k + 1, sizeof(float));
	answer.w = (float*)malloc(sizeof(float)*(k + 1));
	minAnswer.w = (float*)malloc(sizeof(float)*(k + 1));
	answer.foundAnswer = false;

	// Define new mpi type: 'AnswerMPIType'
	int answerBlocklen[4] = { k+1, 1, 1, 1 };
	answerDisp[0] = (char *)&answer.w[0] - (char *)&answer;
	answerDisp[1] = (char *)&answer.q - (char *)&answer;
	answerDisp[2] = (char *)&answer.a - (char *)&answer;
	answerDisp[3] = (char *)&answer.foundAnswer - (char *)&answer;
	MPI_Type_create_struct(4, answerBlocklen, answerDisp, answerType, &AnswerMPIType);
	MPI_Type_commit(&AnswerMPIType);

	// Define new mpi type: 'PointMPIType'
	p.coords = (float*)malloc(k + 1 * sizeof(float));
	int pointBlocklen[2] = { k + 1, 1 };
	pointDisp[0] = (char *)&p.coords[0] - (char *)&p;
	pointDisp[1] = (char *)&p.group - (char *)&p;
	MPI_Type_create_struct(2, pointBlocklen, pointDisp, pointType, &PointMPIType);
	MPI_Type_commit(&PointMPIType);

	// The maximum number of alphas to check
	numOfAlpha = (int) (a_max / a_0);

	if (numOfAlpha >= MAX_ALPHA_CHECK)
	{
		cout << "Wrong number of alpha to check!\n";
		exit(-1);
	}
	alphaSendLeft = numOfAlpha;
	
	// Root send the points to childs
	if (myrank == root)
	{
		sendPoints(&p, points, n, k, numOfProc, &PointMPIType);
	}
	
	// Childs recive the points from root
	else
	{
		recivePoints(&p, points, n, k, numOfProc, &PointMPIType);
	}

	// Root process work
	if (myrank == root)
	{
		arrayOfAlpha = (float*)malloc(numOfAlpha * sizeof(float));
		
		// Initialization 'arrayOfAlpha' with 'numOfAlpha' alpha
		#pragma omp parallel for
		for (i = 0; i < numOfAlpha; i++)
		{
			arrayOfAlpha[i] = a_0 * (i + 1);
		}
		
		// Initialization 'minAnswer.a' with a=1 (maximum) && 'minAnswer.foundAnswer' to false
		minAnswer.a = 1;
		minAnswer.foundAnswer = false;

		// Send alpha to childs
		#pragma omp parallel for
		for (i = 1; i < numOfProc; i++)
		{
			MPI_Send(&arrayOfAlpha[i - 1], 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
			alphaSendLeft--;
		}

		int numOfAnswer = 0;

		for (i=1; i<numOfProc; i++)
		{
			// Recive answer from child
			MPI_Recv(&answer, 1, AnswerMPIType, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			numOfAnswer++;
			
			// If answer is the minimum answer
			if (answer.a < minAnswer.a && answer.foundAnswer)
				initMinAnswer(&answer, &minAnswer, k);

			// Check if the minimum answer is found
			if (numOfAnswer >= numOfProc - 1 && minAnswer.foundAnswer)
			{
				// Tell the child that the work is done
				MPI_Send(&alphaSendLeft, 1, MPI_FLOAT, status.MPI_SOURCE, TERMINATION_TAG, MPI_COMM_WORLD);
			}
			else
			{
				// Send how much alpha still need to check
				MPI_Send(&alphaSendLeft, 1, MPI_FLOAT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
				
				if (alphaSendLeft > 0)
				{
					// Send alpha to child
					MPI_Send(&arrayOfAlpha[numOfAlpha - alphaSendLeft], 1, MPI_FLOAT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
					alphaSendLeft--;

					// Recive answer from child
					MPI_Recv(&answer, 1, AnswerMPIType, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);

					// If answer is the minimum answer
					if (answer.a < minAnswer.a && answer.foundAnswer)
						initMinAnswer(&answer, &minAnswer, k);
				}
			}
		}

		// Write the result to file
		writeToFile(&minAnswer, k);
		
		// Done the runtime check and print the result
		//t2 = MPI_Wtime();
		//cout << "time = " << t2 - t1 << endl;
	}

	// Child process work
	else
	{		
		// Recive alpha from root
		MPI_Recv(&a, 1, MPI_FLOAT, root, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		
		// Child process start the algorithm with his alpha
		for(iteration = 0; iteration < limit; iteration++)
		{
			// Go over all the points
			for (i = 0; i < n, correct; i++)
			{
				result = sign(k, w, points[i]);
				
				// Check if wrong classification
				if ((result == 1 && points[i].group == -1) || (result == -1 && points[i].group == 1))
				{
					correct = false;
					for (int d = 0; d < k + 1; d++)
					{
						w[d] += a * result*(-1)* points[i].coords[d];
					}
				}
			}
			
			// Check if all points are Classifieds well
			if (correct)
			{
				break;
			}
			else
			{
				correct = true;
			}
		}

		numOfWrong = 0;
		// Count number of point P that does not satisfies this criterion
		//#pragma omp parallel for
		for (i = 0; i < n; i++)
		{
			result = sign(k, w, points[i]);
			if ((result == 1 && points[i].group == -1) || (result == -1 && points[i].group == 1))
			{
				numOfWrong++;
			}
		}

		// q check
		q = (float)numOfWrong / n;
		if (q < qc)
		{
			#pragma omp parallel for
			for (int t = 0; t < k + 1; t++)
			{
				answer.w[t] = w[t];
			}
			answer.q = q;
			answer.a = a;
			answer.foundAnswer = true;
		}
		
		// Send the answer to root
		MPI_Send(&answer, 1, AnswerMPIType, root, 0, MPI_COMM_WORLD);

		/* Child process start again the algorithm with his new alpha if needed */
		answer.foundAnswer = false;

		// Recive number of alpha that still need check
		MPI_Recv(&alphaSendLeft, 1, MPI_FLOAT, root, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		// If there is still work to do
		if (status.MPI_TAG != TERMINATION_TAG)
		{
			// W = { 0 }
			#pragma omp parallel for
			for (int d = 0; d < k + 1; d++)
			{
				w[d] = 0;
			}

			if (alphaSendLeft > 0)
			{
				// Recive alpha from root
				MPI_Recv(&a, 1, MPI_FLOAT, root, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

				for (iteration = 0; iteration < limit; iteration++)
				{
					// Go over all the points
					for (i = 0; i < n, correct; i++) 
					{
						result = sign(k, w, points[i]);
						if ((result == 1 && points[i].group == -1) || (result == -1 && points[i].group == 1))
						{
							correct = false;

							#pragma omp parallel for
							for (int d = 0; d < k + 1; d++)
							{
								w[d] += a * result*(-1)* points[i].coords[d];
							}
						}
					}
					// Check if all points are Classifieds well
					if (correct)
					{
						break;
					}
					else
					{
						correct = true;
					}
				}
				numOfWrong = 0;

				// Count number of point P that does not satisfies this criterion
				#pragma omp parallel for
				for (i = 0; i < n; i++)
				{
					result = sign(k, w, points[i]);
					if ((result == 1 && points[i].group == -1) || (result == -1 && points[i].group == 1))
					{
						numOfWrong++;
					}
				}
				
				// q check
				q = (float)numOfWrong / n;
				if (q < qc)
				{
					#pragma omp parallel for
					for (int t = 0; t < k + 1; t++)
					{
						answer.w[t] = w[t];
					}
					answer.q = q;
					answer.a = a;
					answer.foundAnswer = true;
				}
				
				// Send the answer to root
				MPI_Send(&answer, 1, AnswerMPIType, root, 0, MPI_COMM_WORLD);
			}
		}
	}

		MPI_Finalize();
		return 0;
}


int sign(int k, float *w, struct Point p)
{
	float result = 0;

	for (int i = 0; i < k + 1; i++)
	{
		result += w[i] * p.coords[i];
	}

	if (result >= 0)
		return 1;
	else
		return -1;
}

Point* readFromFile(int *n, int *k, float *a_0, float *a_max, int *limit, float *qc, struct Point *points)
{
	FILE *f = fopen(READ_PATH, "r");
	if (f == NULL)
	{
		cout << "Failed to open the text file!\n";
		exit(-1);
	}

	fscanf(f, "%d %d %f %f %d %f", n, k, a_0, a_max, limit, qc);

	if (*n < MIN_POINTS || *n > MAX_POINTS)
	{
		cout << "Wrong number of points!\n";
		exit(-1);
	}

	if (*k >= MAX_DIM)
	{
		cout << "Wrong number of dimensions!\n";
		exit(-1);
	}

	if (*limit > MAX_ITERATION)
	{
		cout << "Wrong number of iteration!\n";
		exit(-1);
	}

	points = (struct Point*)malloc(*n * sizeof(struct Point));

	for (int i = 0; i < *n; i++)
	{
		points[i].coords = (float*)malloc((*k + 1) * sizeof(float));

		// read coords
		for (int j = 0; j < *k; j++)
		{
			fscanf(f, "%f", &points[i].coords[j]);
		}
		points[i].coords[*k] = 1;
		fscanf(f, "%d", &points[i].group);
	}
	fclose(f);
	return points;
}

void writeToFile(struct Answer *minAnswer, int k)
{
	FILE *fw = fopen(WRITE_PATH, "w");
	if (fw == NULL)
	{
		cout << "Failed to open the text file!\n";
		exit(-1);
	}
	if (minAnswer->foundAnswer == true)
	{
		fprintf(fw, "Alpha minimum = %f\t\tq = %f\n", minAnswer->a, minAnswer->q);

		for (int i = 0; i < k; i++)
			fprintf(fw, "%f\n", minAnswer->w[i]);
	}
	else
	{
		fprintf(fw, "Alpha is not found");
	}
	fclose(fw);
}

void sendPoints(struct Point *p, struct Point *points, int n, int k, int numOfProc, MPI_Datatype *PointMPIType)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < k + 1; j++)
			p->coords[j] = points[i].coords[j];

		p->group = points[i].group;

		for (int proc = 1; proc < numOfProc; proc++)
		{
			MPI_Send(p, 1, *PointMPIType, proc, 0, MPI_COMM_WORLD);
		}
	}
}

void recivePoints(struct Point *p, struct Point *points, int n, int k, int numOfProc, MPI_Datatype *PointMPIType)
{
	MPI_Status status;

	for (int i = 0; i < n; i++)
	{
		MPI_Recv(p, 1, *PointMPIType, 0, 0, MPI_COMM_WORLD, &status);

		for (int j = 0; j < k + 1; j++)
			points[i].coords[j] = p->coords[j];

		points[i].group = p->group;
	}
}

void initMinAnswer(struct Answer *answer, struct Answer *minAnswer, int k)
{
	#pragma omp parallel for
	for (int t = 0; t < k + 1; t++)
	{
		minAnswer->w[t] = answer->w[t];
	}
	minAnswer->q = answer->q;
	minAnswer->a = answer->a;
	minAnswer->foundAnswer = answer->foundAnswer;
}