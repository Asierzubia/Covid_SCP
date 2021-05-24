#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define SIZE_WORLD 16
#define PERCENT 0.05
#define MAX_SPEED 2
#define MAX_DIRECTION 7
#define MAX_INCUBATION 5
#define MAX_RECOVERY 10
#define MAX_INFECTION 0.3 //0.6
#define MAX_DEATH 200
#define POPULATION_SIZE 200
#define BATCH 2
#define CUADRICULA 5
#define ITER 1
#define SEED 3


/*Estructura para la parametrizacion*/

typedef struct configuration_t {
    unsigned int world_size;
    unsigned int percent_t;
    unsigned int population_size;
    unsigned int batch;
    unsigned int iter_t;
    unsigned int seed;
} configuration_t;

enum list
{
    INFECTED,
    NOT_INFECTED,
    VACCINED
};

typedef struct index
{

    int id; // index
    enum list l;

} index_t;

typedef struct coord
{
    int x;
    int y;
} coord_t;

typedef struct person
{
    int age;
    int state;
    float prob_infection;
    coord_t coord;
    int speed[2]; // 0 direction 1 speed
    int incubation_period;
    int recovery;
    int id;
    int id_global;
} person_t;

typedef struct person_move
{
    person_t person;
    coord_t coord;
} person_move_t;


void create_person(int procesador);
void per_cicle();
void change_state(person_t person);
void propagate(person_t *person);
person_t *init_lists(int tamano);
void init_gsl();
void change_move_prob(person_t *person);
void change_infection_prob(person_t *person);
void init_world();
void calculate_init_position(person_t *person);
void print_world();
void print_person(person_t person,int procesador);
void calculate_metrics();
void init_person_parameters(person_t *persona,int estado,int id_local);
float calculate_ageMean();
float calculate_prob_death(int edad);
int vacunate(person_t person);
int random_number(int min_num, int max_num);
void realocate_lists();


//Crear DataTypes
void create_data_type_coord(coord_t *coordenadas);
void create_data_type_person(person_t *persona);
void create_data_type_person_move(coord_t *coordenadas, person_t *persona);

// Move
int is_inside_world(int from, int direction, int to_node);
void move_person(person_t *person, int world_rank);
void move(person_t *person, coord_t *coord);

void send_visitors(int flag);
void init_move_list(int size_x, int size_y);
void init_prop_list(int size_x, int size_y);

//Send-Recive
void Psend(int to_node,int flag);
void recive(int flag);


//Mover y propgar los nuevos

void move_arrived();
void propagate_arrived();

coord_t calculate_coord(int x, int y);

// Ficheros
void print_positions();
void save_positions(int world_rank, int iteration);
void print_metrics();
void save_metrics(int world_rank, int iteration);
void calculate_metrics();
char *init_list_archives(int size);
#endif // DEFINITIONS_H