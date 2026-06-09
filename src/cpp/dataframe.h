#ifndef DATAFRAME_H
#define DATAFRAME_H

#include <format>
#include <map>
#include <stdexcept>
#include <sys/_types/_rune_t.h>
#include <variant>
#include <vector>

#include "common.h"

namespace massim {

typedef std::variant<std::vector<bool>, std::vector<int>, std::vector<double>,
                     std::vector<std::string>>
    column_t;


class DataFrame {
  std::map<std::string, const column_t> m_columns;
  int m_num_rows;
  
public:
  DataFrame() : m_num_rows(-1) {}

  void add_int_column(std::string name, const IntArrayType &data) {
    std::vector<int> column(data.begin(), data.end());
    m_columns.insert({name, column});
  }

  void add_double_column(std::string name, const ArrayType &data) {
    std::vector<double> column(data.begin(), data.end());
    m_columns.insert({name, column});
  }

  void add_bool_column(std::string name, const BoolArrayType &data) {
    std::vector<bool> column(data.begin(), data.end());
    m_columns.insert({name, column});
  }

  void add_string_column(std::string name,
                         const std::vector<std::string> &data) {
    m_columns.insert({name, data});
  }

  const column_t &operator[](const std::string &name) const {
    auto it = m_columns.find(name);
    if (it == m_columns.end()) {
      throw std::runtime_error(
          std::format("No column '{}' in dataframe", name));
    }
    return it->second;
  }

  const std::vector<double> &double_column(const std::string &name) const {
    auto &col = (*this)[name];
    if (std::holds_alternative<std::vector<double>>(col)) {
      return std::get<std::vector<double>>(col);
    }
    throw std::runtime_error(
        std::format("Column '{}' does not contain double.", name));
  }

  const std::vector<int> &int_column(const std::string &name) const {
    auto &col = (*this)[name];
    if (std::holds_alternative<std::vector<int>>(col)) {
      return std::get<std::vector<int>>(col);
    }
    throw std::runtime_error(
        std::format("Column '{}' does not contain int.", name));
  }

  const std::vector<bool> &bool_column(const std::string &name) const {
    auto &col = (*this)[name];
    if (std::holds_alternative<std::vector<bool>>(col)) {
      return std::get<std::vector<bool>>(col);
    }
    throw std::runtime_error(
        std::format("Column '{}' does not contain bool.", name));
  }

  const std::vector<std::string> &string_column(const std::string &name) const {
    auto &col = (*this)[name];
    if (std::holds_alternative<std::vector<std::string>>(col)) {
      return std::get<std::vector<std::string>>(col);
    }
    throw std::runtime_error(
        std::format("Column '{}' does not contain string.", name));
  }
};

} // namespace massim

#endif
