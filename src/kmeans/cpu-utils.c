#ifndef CPU_UTILS_C
#define CPU_UTILS_C

#include <stdio.h>
#include <string.h>

#define CREATEDATA_RAND_MAX 10000000

bool isPowerOf2(unsigned int x) {
    return x != 0 && (x & (x - 1)) == 0;
}

template <typename T>
inline bool is_power_of_2(T x) {
    return x != 0 && (x & (x - 1)) == 0;
}

void create_data(float* dataset, uint dataset_size){
	for (int i = 0; i < dataset_size; ++i){
		uint r = rand()%CREATEDATA_RAND_MAX;
		dataset[i] = (float)((r-CREATEDATA_RAND_MAX/2.0)/(float)CREATEDATA_RAND_MAX);
		// printf("%u %f\n",r,dataset[i]);
	}
}

void from_file(const char* filename, 
		uint size,
		uint d,
		float* dset){

	FILE* file = fopen(filename,"r");
	if(file == NULL){
		printf("ERROR: file '%s' does not exist.\n",filename);
		exit(-1);
	}
	uint i = 0;
	while(i < size*d){
		for (uint j = 0; i < size*d-1 && j < d-1; ++j){
			int ret = fscanf(file, "%f,", &dset[i++]);
			if(ret == EOF){
				printf("ERROR: not enough data in file '%s'.\n",filename);
				exit(-1);
			}
		}
		int ret = fscanf(file, "%f", &dset[i++]);
		if(ret == EOF){
			printf("ERROR: not enough data in file '%s'.\n",filename);
			exit(-1);
		}

	}
	fclose(file);
	// printf("%f %f %f %f %f\n",dset[0],dset[1],dset[2],dset[3],dset[4]);
}

void write_data(
		char* file_name,
		uint np,
		uint dim,
		float* points){

	FILE* file = fopen(file_name,"w");
	if(file == NULL){
		printf("ERROR: can't create file.\n");
		return;
	}

	for (int i = 0; i < np; ++i){
		fprintf(file, "%f",points[i*dim]);
		for(int j = 1; j < dim; j++){
			fprintf(file, " %f",points[i*dim+j]);
		}
		fprintf(file,"\n");
	}
	fclose(file);
}

void write_data(
		char* file_name,
		uint np,
		uint dim,
		int* points){

	FILE* file = fopen(file_name,"w");
	if(file == NULL){
		printf("ERROR: can't create file.\n");
		return;
	}

	for (int i = 0; i < np; ++i){
		fprintf(file, "%d",points[i*dim]);
		for(int j = 1; j < dim; j++){
			fprintf(file, " %d",points[i*dim+j]);
		}
		fprintf(file,"\n");
	}
	fclose(file);
}



// Example: cmdOptionExists(argv, argc+argv, "-ntimes")
inline bool cmdOptionExists(char** begin, char** end, const std::string& option){
	return std::find(begin, end, option) != end;
}

inline std::string getStrFromCmdOption(char ** begin, char ** end, const std::string & option){
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return std::string(*itr);
    }
    return "";
}

// Example: getIntFromCmdOption(argv, argc+argv, "-niter", 10)
inline int getIntFromCmdOption(char ** begin, char ** end, const std::string & option, const int default_value){
	char ** itr = std::find(begin, end, option);
	if (itr != end && ++itr != end)
	{
		return atoi(*itr);
	}
	return default_value;
}

int count_lines(FILE *fp) {
    int count = 0;
    int ch;
    int last_char = '\n'; // treat empty file correctly

    while ((ch = fgetc(fp)) != EOF) {
        if (ch == '\n')
            count++;
        last_char = ch;
    }

    if (last_char != '\n' && count > 0)
        count++; // add last line if not newline-terminated
	rewind(fp); // Reset file pointer to the beginning

    return count;
}

int count_cols(FILE *fp) {
	int count = 0;
	int ch;

	// Read until the end of the first line
	while ((ch = fgetc(fp)) != EOF && ch != '\n') {
		if (ch == ',')
			count++;
	}
	count++; // add one for the last column
	rewind(fp); // Reset file pointer to the beginning

	return count;
}

void read_csv(char* file_name, 
		int *num_points,
		int *dim, 
		float **dataset){
	FILE* file = fopen(file_name,"r");
	if(file == NULL){
		printf("ERROR: file '%s' does not exist.\n",file_name);
		exit(-1);
	}

	int nlines = count_lines(file);
	int ncols = count_cols(file);

	printf("INFO: dataset has %d points with dimension %d\n",nlines,ncols);

	//allocate memory for dataset
	*dataset = (float*)malloc(sizeof(float) * nlines * ncols); // Example allocation, adjust size as needed
	if (*dataset == NULL) {
		printf("ERROR: Memory allocation failed.\n");
		exit(-1);
	}

	size_t line_size = 0;
	char* line = NULL;
	int row = 0;
	while(getline(&line, &line_size, file) != -1){
		char* token;
		int col = 0;
		char* tmp = line; // Use line directly for tokenization
		while((token = strsep(&tmp, ",")) != NULL){
			(*dataset)[row*ncols+col] = atof(token); // Correct dereferencing
			col++;
		}
		row++;
	}

	*num_points = nlines;
	*dim = ncols;

	free(line);
	fclose(file);
}

void shuffle_data(float* data, int totalsize, int d, int nshuffle){
	srand(0);
	for (uint i = 0; i < nshuffle; ++i){
		uint r = rand()%totalsize;
		if(r != i){
			for (uint j = 0; j < d; ++j){
				float tmp = data[i*d+j];
				data[i*d+j] = data[r*d+j];
				data[r*d+j] = tmp;
			}
		}
	}
}
#endif // CPU_UTILS_C