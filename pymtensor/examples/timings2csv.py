import csv


def timings2csv(filename):
    with open(filename, 'r') as f1, open('timings.csv', 'w') as f2:
        writer = csv.writer(f2)
        for line in f1:
            if '{' in line:
                tensor_name = line.split()[0]
                writer.writerow([])
                writer.writerow([tensor_name])
                writer.writerow(['class', 'time (s)', 'rows', 'free vars'])
            elif 'Crystal class' in line:
                crystal_class = line.split()[-1]
                time = 0e0
                num_free_vars = 0
                rows = 0
                # Get the rest of this crystal class' data
                for subline in f1:
                    if 'Number of unknowns' in subline:
                        num_free_vars = int(subline.split()[-1])
                    if 'Rotate tensor' in subline:
                        time += float(subline.split()[-2])
                    if 'Solve linear system' in subline:
                        time += float(subline.split()[-2])
                    if 'len(eqs)' in subline:
                        rows = int(subline.split()[-1])
                    if 'Number of free variables' in subline:
                        # This overwrites the value from 'Number of unknowns'
                        num_free_vars = int(subline.split()[-1])
                    if 'To full solution' in subline:
                        time += float(subline.split()[-2])
                        break
                data = [crystal_class, time, rows, num_free_vars]
                writer.writerow(data)
                

if __name__ == '__main__':
    timings2csv('results.txt')