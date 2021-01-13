# Share ECR Container Across multiple accounts

1.  Open the [Amazon ECR console](https://console.aws.amazon.com/ecr/) for your primary account.

2.  Select the name of the repository that you want to modify.

3.  From the navigation menu, choose Permissions.

4.  To add a repository policy for your secondary account from within your primary account, choose Edit policy JSON, enter your policy into the code editor, and then choose Save.

Important: In your policy, include the account number of the secondary account and the actions that the account can perform against the repository.

## [Add AWS AccountID as Principal](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_principal.html)

For example, given an account ID of 123456789012, you can use either of the following methods to specify that account in the Principal element:

```json
"Principal": { "AWS": "arn:aws:iam::123456789012:root" }
```

```json
"Principal": { "AWS": "123456789012" }
```

## Policy Document to get ECR Container

The following example repository policy allows a any account to pull images:

```json
{
  "Version": "2008-10-17",
  "Statement": [
    {
      "Sid": "Allow External Download",
      "Effect": "Allow",
      "Principal": "*",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer"
      ]
    }
  ]
}
```
