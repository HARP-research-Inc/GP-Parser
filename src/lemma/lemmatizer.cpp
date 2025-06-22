#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

int main(int argc, char **argv) {
  // Load the lemma lookup table
  std::unordered_map<std::string, std::string> lemma_map;
  
  std::ifstream file("wordnet-lemma-lookup.txt");
  std::string line;
  
  while (std::getline(file, line)) {
    size_t tab_pos = line.find('\t');
    if (tab_pos != std::string::npos) {
      std::string word = line.substr(0, tab_pos);
      std::string lemma = line.substr(tab_pos + 1);
      lemma_map[word] = lemma;
    }
  }
  
  std::cout << "Loaded " << lemma_map.size() << " lemma mappings\n";
  
  // Process input
  std::string token;
  while (std::getline(std::cin, token)) {
    auto it = lemma_map.find(token);
    if (it != lemma_map.end()) {
      std::cout << it->second << "\n";
    } else {
      std::cout << token << "\n";
    }
  }
  
  return 0;
} 