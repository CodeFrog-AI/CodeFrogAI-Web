import { PageContainer } from "@/components/layout/PageContainer";
import { EmptyState } from "@/components/ui/EmptyState";

export function PullRequestsPage() {
  return (
    <PageContainer title="Pull Requests" description="Pull requests CodeFrog has opened or reviewed for your repository.">
      <EmptyState
        icon="pull-requests"
        title="No pull requests yet"
        description="Your pull requests will be listed here once GitHub is connected and CodeFrog has opened or reviewed some."
      />
    </PageContainer>
  );
}
