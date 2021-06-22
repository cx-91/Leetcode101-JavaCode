
# Homework 1: Beware of Geeks Bearing Gift Cards

## Introduction

Do the medium problems

## Part 1: Binary Search

Be sure to make the repository **private**.

The next step is to set up Travis CI. To do this, go to
https://travis-ci.org and authorize Travis to link with your GitHub


```
examplefile.gft giftcardexamplewriter.c giftcard.h giftcardreader.c Makefile
```


1. *Two* test cases, `crash1.gft` and `crash2.gft`, that cause the
   program to crash; each crash should have a different root cause.
2. One test case, `hang.gft`, that causes the program to loop
   infinitely. (Hint: you may want to examine the "program" record type
   to find a bug that causes the program to loop infinitely.)
3. A text file, `bugs.txt` explaining the bug triggered by each of your
   three test cases. 

To create your own test files, you may want to look at
`giftcardexamplewriter.c`. Although it is no better written than
`giftcardreader.c`, it should help you understand the basics of the file
format and how to produce gift card files.

Finally, fix the bugs that are triggered by your test cases, and verify
that the program no longer crashes / hangs on your test cases. To make
sure that these bugs don't come up again as the code evolves, have
Travis automatically build and run the program on your test suite.

## Sting Manipulate

        for(int i = 0; i < paragraph.length();){
            StringBuilder sb = new StringBuilder();
            while(i < paragraph.length() && Character.isLetter(Character.toLowerCase(paragraph.charAt(i)))){
                sb.append(Character.toLowerCase(paragraph.charAt(i)));
                i++;
            }
            
            while(i < paragraph.length() && !Character.isLetter(paragraph.charAt(i))){
                i++;
            }
        }