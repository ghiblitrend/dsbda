#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

int main() {
    ifstream file("sample_weather.txt");
    if (!file) {
        cerr << "Error opening file.\n";
        return 1;
    }

    string line;
    getline(file, line); // skip the header line

    double totalTemp = 0.0, totalDew = 0.0, totalWind = 0.0;
    int count = 0;

    while (getline(file, line)) {
        string date;
        double temp, dew, wind;

        stringstream ss(line);
        ss >> date >> temp >> dew >> wind;

        totalTemp += temp;
        totalDew += dew;
        totalWind += wind;
        count++;
    }

    if (count == 0) {
        cout << "No data to process.\n";
        return 0;
    }

    cout << "Average Temperature: " << totalTemp / count << "°C\n";
    cout << "Average Dew Point: " << totalDew / count << "°C\n";
    cout << "Average Wind Speed: " << totalWind / count << " km/h\n";

    file.close();
    return 0;
}

