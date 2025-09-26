#include "distribution.h"

#include <cctype>
#include <regex>
#include <variant>
#include <vector>

namespace massim {

class ParseException : public MassimException {

public:
  ParseException(const std::string &msg) : MassimException(msg) {}
};

enum TOK_TYPE {
  NONE,
  NAME_TOK,
  NUMBER,
  OPEN_PAREN,
  CLOSE_PAREN,
  COMMA,
  BINOP,
  UNOP,
  DISTRIB,
  UNKNOWN,
  END_TOK

};

struct Token {
  TOK_TYPE ttype;
  std::variant<std::string, double, char, DistUPtr> data;
};

std::string tok2str(const Token &t) {
  std::string ttype;
  std::string tdat;
  switch (t.ttype) {
  case NONE:
    ttype = "NONE";
    break;
  case NAME_TOK:
    ttype = "NAME_TOK";
    tdat = std::get<std::string>(t.data);
    break;
  case NUMBER:
    ttype = "NUMBER";
    tdat = std::get<double>(t.data);
    break;
  case OPEN_PAREN:
    ttype = "(";
    break;
  case CLOSE_PAREN:
    ttype = ")";
    break;
  case COMMA:
    ttype = ",";
    break;
  case BINOP:
    ttype = "BINOP";
    tdat = std::get<char>(t.data);
    break;
  case UNOP:
    ttype = "UNOP";
    tdat = std::get<char>(t.data);
    break;
  case DISTRIB:
    ttype = "DISTRIB";
    tdat = std::get<DistUPtr>(t.data)->name();
    break;
  case UNKNOWN:
    ttype = "UNKNOWN";
    break;
  case END_TOK:
    ttype = "END_TOK";
    break;
  }
  return ttype + ": " + tdat;
}

// Split into words, numbers, or punct (parens, dot, arithmentic)
const std::regex
    TOK_RE(R"([a-zA-Z_]+|(?:[-+]?\d*\.?\d+)(?:[eE](?:[-+]?\d+))?|.)");

std::unique_ptr<Distribution> gen_dist(const std::string &dist_type,
                                       const std::vector<double> &args) {
  if (dist_type == "constant") {
    if (args.size() != 1) {
      throw MassimException("Constant distribution requires 1 argument");
    }
    return std::make_unique<ConstantDistribution>(args[0]);
  }
  if (dist_type == "uniform") {
    if (args.size() != 2) {
      throw MassimException("Uniform distribution requires two arguments");
    }
    return std::make_unique<UniformDistribution>(args[0], args[1]);
  }
  if (dist_type == "normal") {
    if (args.size() != 2) {
      throw MassimException("Normal distribution requires two arguments");
    }
    return std::make_unique<NormalDistribution>(args[0], args[1]);
  }
  if (dist_type == "lognormal") {
    if (args.size() != 2) {
      throw MassimException("Lognormal distribution requires two arguments");
    }
    return std::make_unique<LognormalDistribution>(args[0], args[1]);
  }
  if (dist_type == "linspace") {
    if (args.size() != 2) {
      throw MassimException("Linspace distribution requires two arguments");
    }
    return std::make_unique<LinspaceDistribution>(args[0], args[1]);
  }

  throw MassimException("Unknown distribution: " + dist_type);
}

std::vector<Token> tokenize_dist(const std::string &txt) {
  auto b = std::sregex_token_iterator(txt.begin(), txt.end(), TOK_RE);
  auto rend = std::sregex_token_iterator();

  std::vector<Token> toks;
  TOK_TYPE last_tok;

  std::vector<std::string> result;
  for (; b != rend; ++b) {
    const std::string &part = *b;

    char c = part[0];
    if (c == ' ')
      continue;
    if (c == '(')
      toks.push_back(Token{OPEN_PAREN});
    else if (c == ')')
      toks.push_back(Token{CLOSE_PAREN});
    else if (c == ',')
      toks.push_back(Token{COMMA});
    else if (part.length() == 1 && strchr("+-*/", c))
      toks.push_back(Token{BINOP, c});
    else {
      if (c == '-' || c == '+') {
        // We are numeric, but the sign may have been associated with the
        // number instead of with arithmetic
        if (last_tok == NONE || last_tok == OPEN_PAREN || last_tok == COMMA) {
          toks.push_back(Token{NUMBER, std::stod(part)});
        } else {
          toks.push_back(Token{BINOP, part[0]});
          toks.push_back(Token{NUMBER, std::stod(part.substr(1))});
        }
      } else if (std::isalpha(c)) {
        toks.push_back(Token{NAME_TOK, part});
      } else if (std::isdigit(c)) {
        toks.push_back(Token{NUMBER, std::stod(part)});
      } else {
        std::string msg = "Unknown token '";
        msg.push_back(c);
        msg += "' in text";
        throw ParseException(msg);
      }
    }

    last_tok = toks.back().ttype;
  }

  for (const auto &tok : toks) {
    switch (tok.ttype) {
    case NAME_TOK:
      result.push_back("D:" + std::get<std::string>(tok.data));
      break;
    case OPEN_PAREN:
      result.push_back("(");
      break;
    case CLOSE_PAREN:
      result.push_back(")");
      break;
    case COMMA:
      result.push_back(",");
      break;
    case NUMBER:
      result.push_back(std::to_string(std::get<double>(tok.data)));
      break;
    case BINOP: {
      char c = std::get<char>(tok.data);
      std::string msg = "A:";
      msg.push_back(c);
      result.push_back(msg);
    } break;
    default:
      result.push_back("Idunno");
    }
  }
  toks.push_back(Token{END_TOK});
  return toks;
}

struct DistNode {
  std::unique_ptr<Distribution> dist;
  std::vector<double> args;
  std::unique_ptr<DistNode> lhs;
  std::unique_ptr<DistNode> rhs;
};

int get_prec(const Token &t) {

  switch (t.ttype) {
  case OPEN_PAREN:
    return 999;

  case BINOP: {
    char opchar = std::get<char>(t.data);
    switch (opchar) {
    case '*':
    case '/':
      return 10;
    case '+':
    case '-':
      return 11;
    default:
      throw ParseException(std::format("Unexpected operator: {}", opchar));
    }
  }
  case UNOP: {
    char opchar = std::get<char>(t.data);
    switch (opchar) {
    case '-':
      return 5;
    default:
      throw ParseException(std::format("Unexpected operator: {}", opchar));
    }
  }
  default:
    throw ParseException(std::format("Unexpected token: {}", tok2str(t)));
  }
}

std::unique_ptr<Distribution> parse_dist(const std::string &dist) {
  std::vector<Token> toks = tokenize_dist(dist);

  auto cur_tok = toks.begin();
  Token last_tok;
  std::vector<std::unique_ptr<Distribution>> stack;

  auto op_char = [&]() { return std::get<char>(cur_tok->data); };

  auto expect = [&](TOK_TYPE ttype, const std::string &err) {
    cur_tok++;
    if (cur_tok->ttype != ttype) {
      throw ParseException(err);
    }
  };

  auto expect_arg = [&]() {
    cur_tok++;
    if (cur_tok->ttype != NUMBER) {
      throw ParseException("Expected numeric argument");
    }
    double val = std::get<double>(cur_tok->data);
    return val;
  };

  auto consume_if = [&](TOK_TYPE ttype) {
    auto next = cur_tok + 1;
    if (next->ttype == ttype) {
      cur_tok = next;
      return true;
    }
    return false;
  };

  // First, we want to translate our infix expression into a postfix one.
  // All operands will be distributions, so raw numbers are converted
  // to constants dists first.
  // We use Djikstra's shunting yard method.

  std::vector<Token> opstack;
  std::vector<Token> postfix;
  // unary_mode keeps track of when a '-' operator can be unary
  bool unary_mode = true;

  while (cur_tok->ttype != END_TOK) {

    if (cur_tok->ttype == NAME_TOK) {
      // An operand; parse and place in postfix output
      std::string dist_name = std::get<std::string>(cur_tok->data);
      std::vector<double> args;
      expect(OPEN_PAREN,
             std::format("Error parsing distribution {}. Expected '('.",
                         dist_name));
      // All our dists take at least one argument
      while (true) {
        args.push_back(expect_arg());
        if (consume_if(CLOSE_PAREN))
          break;
        expect(COMMA,
               std::format("Error parsing distribution {}. Expected ','.",
                           dist_name));
      }
      postfix.push_back(Token{DISTRIB, gen_dist(dist_name, args)});
      unary_mode = false;
    } else if (cur_tok->ttype == NUMBER) {
      // Numbers get converted to constant distributions; hence operands
      double val = std::get<double>(cur_tok->data);
      postfix.push_back(
          Token{DISTRIB, std::make_unique<ConstantDistribution>(val)});
      unary_mode = false;
    } else if (cur_tok->ttype == OPEN_PAREN) {
      opstack.push_back(std::move(*cur_tok));
      unary_mode = true;
    } else if (cur_tok->ttype == BINOP) {
      if (unary_mode && op_char() == '-') {
        opstack.push_back(Token{UNOP, '-'});
        unary_mode = false;
      } else {
        int prec = get_prec(std::move(*cur_tok));
        while (!opstack.empty() && get_prec(opstack.back()) <= prec) {
          postfix.push_back(std::move(opstack.back()));
          opstack.pop_back();
        }
        opstack.push_back(std::move(*cur_tok));
        unary_mode = true;
      }
    } else if (cur_tok->ttype == CLOSE_PAREN) {
      while (!opstack.empty() && opstack.back().ttype != OPEN_PAREN) {
        postfix.push_back(std::move(opstack.back()));
        opstack.pop_back();
      }
      if (opstack.empty())
        throw ParseException("Extra closing parenthesis");
      opstack.pop_back();
    } else {
      throw ParseException(
          std::format("Unexpected token: {}", tok2str(*cur_tok)));
    }
    cur_tok++;
  }
  while (!opstack.empty()) {
    postfix.push_back(std::move(opstack.back()));
    opstack.pop_back();
  }

  // Now our formula is in postfix mode, and should only contain operator
  // and distribution tokens. Convert to a distribution.
  std::vector<DistUPtr> dist_stack;
  for (cur_tok = postfix.begin(); cur_tok != postfix.end(); ++cur_tok) {
    if (cur_tok->ttype == DISTRIB) {
      dist_stack.push_back(std::move(std::get<DistUPtr>(cur_tok->data)));
    } else if (cur_tok->ttype == UNOP) {
      if (dist_stack.empty())
        throw ParseException("Missing argument for unary operator");
      auto topdist = std::move(dist_stack.back());
      dist_stack.pop_back();
      DistUPtr newdist = std::make_unique<UnOpDistribution>(
          std::move(topdist), std::get<char>(cur_tok->data));
      dist_stack.push_back(std::move(newdist));
    } else if (cur_tok->ttype == BINOP) {
      if (dist_stack.size() < 2)
        throw ParseException("Missing arguments for binary operator");
      auto rhs = std::move(dist_stack.back());
      dist_stack.pop_back();
      auto lhs = std::move(dist_stack.back());
      dist_stack.pop_back();
      DistUPtr newdist = std::make_unique<BinOpDistribution>(
          std::move(lhs), std::move(rhs), std::get<char>(cur_tok->data));
      dist_stack.push_back(std::move(newdist));
    } else {
      throw ParseException(
          std::format("Unexpected token '{}' in transformed expression;",
                      tok2str(*cur_tok)));
    }
  }

  if (dist_stack.size() != 1) {
    throw ParseException("Unparseable expression; too many operands.");
  }

  return std::move(dist_stack.back());
}

} // namespace massim
